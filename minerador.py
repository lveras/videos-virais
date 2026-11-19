"""Minerador de vídeos virais de referência, extrator ASMR e indexador SQLite."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

import db

BASE_DIR = Path(__file__).resolve().parent
SEEDS_PATH = BASE_DIR / "seeds_reels.json"

REQUIRED_DIRS = [
    BASE_DIR / "data",
    BASE_DIR / "baixados",
    BASE_DIR / "criados",
    BASE_DIR / "referencias" / "audios_asmr",
    BASE_DIR / "referencias" / "frames",
    BASE_DIR / "output",
    BASE_DIR / "workflows",
]


def ensure_directories() -> None:
    """Garante que todas as pastas do pipeline existam."""
    for folder in REQUIRED_DIRS:
        folder.mkdir(parents=True, exist_ok=True)


def run_cmd(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    """Executa um comando externo capturando saída em UTF-8."""
    return subprocess.run(
        cmd,
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def infer_shortcode_from_url(url: str) -> str:
    """Extrai o identificador/shortcode de URLs do Instagram, YouTube Shorts ou TikTok."""
    patterns = [
        r"instagram\.com/reels?/([A-Za-z0-9_-]+)",
        r"youtube\.com/shorts/([A-Za-z0-9_-]+)",
        r"youtu\.be/([A-Za-z0-9_-]+)",
        r"v=([A-Za-z0-9_-]+)",
        r"tiktok\.com/@[^/]+/video/(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", url.rstrip("/").split("/")[-1])
    return cleaned or "video_ref"


def fetch_metadata(url: str) -> dict[str, Any]:
    """Coleta metadados JSON do vídeo via yt-dlp sem baixar o arquivo."""
    res = run_cmd(["yt-dlp", "--no-update", "--dump-json", "--no-download", url])
    if res.returncode != 0 or not res.stdout.strip():
        return {}
    try:
        # yt-dlp pode emitir uma linha JSON por vídeo; pega a primeira válida.
        for line in res.stdout.splitlines():
            line = line.strip()
            if line.startswith("{"):
                return json.loads(line)
    except json.JSONDecodeError:
        return {}
    return {}


def download_video(url: str, dest_path: Path) -> tuple[bool, str]:
    """Baixa o vídeo em MP4 via yt-dlp caso ainda não exista localmente."""
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return True, "ja_existente"

    base_cmd = [
        "yt-dlp",
        "--no-update",
        "-f",
        "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format",
        "mp4",
        "-o",
        str(dest_path),
    ]
    res = run_cmd(base_cmd + [url])
    if res.returncode != 0 or not dest_path.exists() or dest_path.stat().st_size == 0:
        if "youtube.com" in url or "youtu.be" in url:
            res_android = run_cmd(
                base_cmd + ["--extractor-args", "youtube:player_client=android", url]
            )
            if (
                res_android.returncode == 0
                and dest_path.exists()
                and dest_path.stat().st_size > 0
            ):
                return True, "baixado_android"
            err = (res_android.stderr or res_android.stdout or res.stderr or "erro desconhecido").strip()
            return False, err
        err = (res.stderr or res.stdout or "erro desconhecido").strip()
        return False, err
    return True, "baixado"


def probe_media(video_path: Path) -> dict[str, Any]:
    """Inspeciona resolução, duração, FPS e presença de áudio usando ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]
    res = run_cmd(cmd)
    if res.returncode != 0 or not res.stdout.strip():
        return {"duration_sec": 0.0, "width": 0, "height": 0, "fps": 0.0, "has_audio": False}

    info = json.loads(res.stdout)
    streams = info.get("streams", [])
    fmt = info.get("format", {})

    video_stream = next((s for s in streams if s.get("codec_type") == "video"), {})
    has_audio = any(s.get("codec_type") == "audio" for s in streams)

    duration_raw = video_stream.get("duration") or fmt.get("duration") or 0.0
    try:
        duration_sec = round(float(duration_raw), 3)
    except (TypeError, ValueError):
        duration_sec = 0.0

    width = int(video_stream.get("width") or 0)
    height = int(video_stream.get("height") or 0)

    fps_str = video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate") or "0/1"
    fps = 0.0
    try:
        if "/" in str(fps_str):
            num, den = str(fps_str).split("/", 1)
            if int(den) != 0:
                fps = round(float(Fraction(int(num), int(den))), 2)
        else:
            fps = round(float(fps_str), 2)
    except (ValueError, ZeroDivisionError):
        fps = 0.0

    return {
        "duration_sec": duration_sec,
        "width": width,
        "height": height,
        "fps": fps,
        "has_audio": has_audio,
    }


def extract_and_master_audio(
    video_path: Path, raw_wav: Path, asmr_wav: Path, has_audio: bool
) -> tuple[str | None, str | None]:
    """Extrai o áudio bruto em WAV e gera versão masterizada ASMR via FFmpeg."""
    if not has_audio:
        return None, None

    cmd_raw = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "44100",
        "-ac",
        "2",
        str(raw_wav),
    ]
    res_raw = run_cmd(cmd_raw)
    if res_raw.returncode != 0 or not raw_wav.exists():
        print(f"  [AVISO] Falha ao extrair audio bruto de {video_path.name}: {res_raw.stderr.strip()}")
        return None, None

    cmd_asmr = [
        "ffmpeg",
        "-y",
        "-i",
        str(raw_wav),
        "-af",
        "highpass=f=60,treble=g=4:f=4500,acompressor=threshold=-18dB:ratio=3:attack=5:release=50,loudnorm=I=-14:TP=-1.5:LRA=11",
        str(asmr_wav),
    ]
    res_asmr = run_cmd(cmd_asmr)
    if res_asmr.returncode != 0 or not asmr_wav.exists():
        print(f"  [AVISO] Falha ao masterizar ASMR de {video_path.name}: {res_asmr.stderr.strip()}")
        return raw_wav.relative_to(BASE_DIR).as_posix(), None

    return (
        raw_wav.relative_to(BASE_DIR).as_posix(),
        asmr_wav.relative_to(BASE_DIR).as_posix(),
    )


def extract_keyframes(video_path: Path, shortcode: str, duration_sec: float) -> list[str]:
    """Extrai 3 frames-chave em 15%, 50% e 85% da duração do vídeo."""
    frames_dir = BASE_DIR / "referencias" / "frames"
    ratios = [0.15, 0.50, 0.85]
    saved: list[str] = []

    effective_duration = duration_sec if duration_sec > 0.5 else 3.0
    for idx, ratio in enumerate(ratios, start=1):
        ts = max(0.1, round(effective_duration * ratio, 2))
        out_jpg = frames_dir / f"{shortcode}_f{idx}.jpg"
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{ts:.2f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(out_jpg),
        ]
        res = run_cmd(cmd)
        if res.returncode == 0 and out_jpg.exists():
            saved.append(out_jpg.relative_to(BASE_DIR).as_posix())
    return saved


def process_item(item: dict[str, Any]) -> bool:
    """Processa um vídeo completo: metadados, download, ffprobe, áudio ASMR, frames e DB."""
    url = item["url"].strip()
    shortcode = (item.get("shortcode") or infer_shortcode_from_url(url)).strip()
    categoria = item.get("categoria_fisica") or "satisfying_generic"
    descricao = item.get("descricao_visual") or ""

    print(f"\n==> Processando [{shortcode}] ({categoria})")
    print(f"    URL: {url}")

    meta = fetch_metadata(url)
    if not meta and not item.get("shortcode"):
        pass
    elif meta.get("id") and not item.get("shortcode"):
        shortcode = str(meta["id"])

    video_path = BASE_DIR / "baixados" / f"{shortcode}.mp4"
    ok, status = download_video(url, video_path)
    if not ok:
        print(f"    [ERRO] Falha no download de {shortcode}: {status}")
        return False

    print(f"    Video ({status}): {video_path.relative_to(BASE_DIR).as_posix()}")

    probe = probe_media(video_path)
    raw_wav = BASE_DIR / "referencias" / "audios_asmr" / f"{shortcode}_raw.wav"
    asmr_wav = BASE_DIR / "referencias" / "audios_asmr" / f"{shortcode}_asmr.wav"
    raw_rel, asmr_rel = extract_and_master_audio(
        video_path, raw_wav, asmr_wav, probe["has_audio"]
    )
    frames_rel = extract_keyframes(video_path, shortcode, probe["duration_sec"])

    uploader = meta.get("uploader") or meta.get("channel") or meta.get("uploader_id")
    caption = meta.get("description") or meta.get("title") or ""
    likes = meta.get("like_count")
    comments = meta.get("comment_count")

    db.upsert_video_referencia(
        {
            "shortcode": shortcode,
            "url": url,
            "uploader": uploader,
            "caption": caption,
            "categoria_fisica": categoria,
            "descricao_visual": descricao or caption,
            "likes": int(likes) if likes is not None else None,
            "comments": int(comments) if comments is not None else None,
            "duration_sec": probe["duration_sec"],
            "width": probe["width"],
            "height": probe["height"],
            "fps": probe["fps"],
            "video_path": video_path.relative_to(BASE_DIR).as_posix(),
            "audio_raw_path": raw_rel,
            "audio_asmr_path": asmr_rel,
            "frames_json": frames_rel,
        }
    )
    print(
        f"    OK: {probe['width']}x{probe['height']} @ {probe['fps']}fps | "
        f"{probe['duration_sec']}s | audio_asmr={'SIM' if asmr_rel else 'NAO'} | "
        f"frames={len(frames_rel)}"
    )
    return True


def normalize_channel_shorts_url(channel_ref: str) -> str:
    """Normaliza handle (@Canal) ou URL do YouTube para a aba /shorts."""
    ref = channel_ref.strip().rstrip("/")
    if not ref.startswith(("http://", "https://")):
        handle = ref if ref.startswith("@") else f"@{ref}"
        return f"https://www.youtube.com/{handle}/shorts"
    if "youtube.com/" in ref and not ref.endswith("/shorts"):
        return f"{ref}/shorts"
    return ref


def fetch_channel_top_shorts(
    channel_ref: str, top: int, categoria: str, descricao: str
) -> list[dict[str, Any]]:
    """Lista os Shorts mais vistos de um canal do YouTube usando yt-dlp."""
    channel_url = normalize_channel_shorts_url(channel_ref)
    pool_limit = max(top * 5, 25)
    print(f"Buscando top {top} Shorts de {channel_url} (amostra: {pool_limit})...")
    res = run_cmd(
        [
            "yt-dlp",
            "--no-update",
            "--flat-playlist",
            "--dump-json",
            "--playlist-end",
            str(pool_limit),
            channel_url,
        ]
    )
    if res.returncode != 0 or not res.stdout.strip():
        print(f"[ERRO] Nao foi possivel listar Shorts de {channel_url}: {res.stderr.strip()}")
        return []

    entries: list[dict[str, Any]] = []
    for line in res.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    entries.sort(key=lambda e: int(e.get("view_count") or 0), reverse=True)
    selected = entries[:top]

    inferred_cat = categoria
    if inferred_cat == "satisfying_custom":
        lower_url = channel_url.lower()
        if "hydraulic" in lower_url or "press" in lower_url:
            inferred_cat = "hydraulic_press_crush"
        elif "rug" in lower_url or "carpet" in lower_url:
            inferred_cat = "rug_cleaning_asmr"

    items: list[dict[str, Any]] = []
    for entry in selected:
        vid = str(entry.get("id") or "").strip()
        if not vid:
            continue
        items.append(
            {
                "shortcode": vid,
                "url": entry.get("url") or f"https://www.youtube.com/shorts/{vid}",
                "categoria_fisica": inferred_cat,
                "descricao_visual": descricao or entry.get("title") or "",
            }
        )
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minerador de videos virais ASMR/Satisfying para a Base de Conhecimento."
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="URL unica (Instagram Reel, YouTube Shorts, etc.) para adicionar à base.",
    )
    parser.add_argument(
        "--channel-shorts",
        type=str,
        default=None,
        help="URL ou @handle de canal do YouTube para minerar os Shorts mais vistos.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=3,
        help="Quantidade de Shorts mais vistos a minerar com --channel-shorts (padrao: 3).",
    )
    parser.add_argument(
        "--categoria",
        type=str,
        default="satisfying_custom",
        help="Categoria fisica/mecanica do video quando usado com --url ou --channel-shorts.",
    )
    parser.add_argument(
        "--descricao",
        type=str,
        default="",
        help="Descricao visual da acao fisica quando usado com --url ou --channel-shorts.",
    )
    parser.add_argument(
        "--seeds",
        type=Path,
        default=SEEDS_PATH,
        help="Caminho do JSON de sementes iniciais (padrao: seeds_reels.json).",
    )
    return parser.parse_args()


def main() -> int:
    ensure_directories()
    db.init_db()
    args = parse_args()

    if args.channel_shorts:
        items = fetch_channel_top_shorts(
            args.channel_shorts, args.top, args.categoria, args.descricao
        )
        if not items:
            return 1
    elif args.url:
        items = [
            {
                "shortcode": infer_shortcode_from_url(args.url),
                "url": args.url,
                "categoria_fisica": args.categoria,
                "descricao_visual": args.descricao,
            }
        ]
    else:
        if not args.seeds.exists():
            print(f"[ERRO] Arquivo de sementes nao encontrado: {args.seeds}", file=sys.stderr)
            return 1
        items = json.loads(args.seeds.read_text(encoding="utf-8"))

    success_count = 0
    failed_items: list[str] = []

    for item in items:
        if process_item(item):
            success_count += 1
        else:
            failed_items.append(item.get("shortcode") or item.get("url"))

    print(f"\n=== Resumo da Mineracao: {success_count}/{len(items)} videos processados ===")
    if failed_items:
        print(f"Falhas: {', '.join(failed_items)}")
    if success_count == 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
