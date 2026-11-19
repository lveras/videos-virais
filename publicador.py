"""Etapa final da esteira: validação FFprobe e empacotamento multi-rede com cross-linking."""

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
CRIADOS_DIR = BASE_DIR / "criados"
OUTPUT_DIR = BASE_DIR / "output"
PRONTOS_DIR = OUTPUT_DIR / "prontos"
PUBLICADOS_DIR = OUTPUT_DIR / "publicados"
CONFIG_DIR = BASE_DIR / "config"
DEFAULT_HANDLE = "@ASMRViralLab"
DEFAULT_OUT_PREFIX = "pacote_publicacao_4_videos"


def ensure_publishing_structure() -> None:
    """Garante as pastas organizadas de saída e configuração de credenciais."""
    for folder in (OUTPUT_DIR, PRONTOS_DIR, PUBLICADOS_DIR, CONFIG_DIR):
        folder.mkdir(parents=True, exist_ok=True)
    example_cfg = CONFIG_DIR / "credenciais_publicacao.example.json"
    if not example_cfg.exists():
        example_cfg.write_text(
            json.dumps(
                {
                    "_instrucoes": "Copie este arquivo para config/credenciais_publicacao.json e preencha as credenciais.",
                    "youtube_shorts": {
                        "enabled": False,
                        "client_secrets_file": "config/youtube_client_secrets.json",
                        "token_file": "config/youtube_token.json",
                    },
                    "instagram_reels": {
                        "enabled": False,
                        "ig_user_id": "SEU_IG_USER_ID_PROFISSIONAL",
                        "access_token": "SEU_META_GRAPH_LONG_LIVED_TOKEN",
                    },
                    "tiktok": {
                        "enabled": False,
                        "client_key": "SEU_TIKTOK_CLIENT_KEY",
                        "client_secret": "SEU_TIKTOK_CLIENT_SECRET",
                    },
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )


def normalize_handle(handle: str) -> str:
    """Garante que o handle comece com @ e não tenha espaços."""
    cleaned = handle.strip()
    if not cleaned:
        return DEFAULT_HANDLE
    return cleaned if cleaned.startswith("@") else f"@{cleaned}"


def parse_hashtags(raw_hashtags: str | None) -> list[str]:
    """Extrai, normaliza e remove duplicatas de hashtags mantendo a ordem."""
    tokens = re.findall(r"#?[A-Za-z0-9_À-ÿ]+", raw_hashtags or "")
    seen: set[str] = set()
    result: list[str] = []
    for token in tokens:
        tag = token if token.startswith("#") else f"#{token}"
        key = tag.lower()
        if key not in seen:
            seen.add(key)
            result.append(tag)
    for fallback in ("#ASMR", "#OddlySatisfying", "#Satisfying"):
        if fallback.lower() not in seen:
            seen.add(fallback.lower())
            result.append(fallback)
    return result


ORIGINAL_VIDEO_MAP: dict[str, str] = {
    "FDa1Oa2y0NY": "criado_01_prensa_cristal_esmeralda.mp4",
    "DfNRXxjou-k": "criado_02_corte_capsula_iridescente.mp4",
    "Df6LIe7uxxw": "criado_03_torno_helicoidal_damasco.mp4",
    "uhQXBMZ-sTg": "criado_04_rodo_espuma_prismatica.mp4",
}


def resolve_final_video(shortcode: str) -> Path | None:
    """Localiza o MP4 final em criados/ priorizando vídeos originais e _final_viral.mp4."""
    candidates: list[Path] = []
    if shortcode in ORIGINAL_VIDEO_MAP:
        candidates.append(CRIADOS_DIR / ORIGINAL_VIDEO_MAP[shortcode])
    candidates.extend(
        [
            CRIADOS_DIR / f"{shortcode}_final_viral.mp4",
            CRIADOS_DIR / f"{shortcode}_preview_asmr_remix.mp4",
            OUTPUT_DIR / f"{shortcode}_final_viral.mp4",
            OUTPUT_DIR / f"{shortcode}_preview_asmr_remix.mp4",
        ]
    )
    for path in candidates:
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


def probe_final_video(video_path: Path) -> dict[str, Any]:
    """Valida via ffprobe se o MP4 final atende 1080x1920, H.264 e áudio AAC."""
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
    res = subprocess.run(
        cmd,
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if res.returncode != 0 or not res.stdout.strip():
        raise RuntimeError(f"ffprobe falhou para {video_path}: {res.stderr.strip()}")

    info = json.loads(res.stdout)
    streams = info.get("streams", [])
    fmt = info.get("format", {})

    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if not video_stream:
        raise RuntimeError(f"Nenhuma faixa de vídeo encontrada em {video_path}")
    if not audio_stream:
        raise RuntimeError(f"Nenhuma faixa de áudio encontrada em {video_path}")

    width = int(video_stream.get("width") or 0)
    height = int(video_stream.get("height") or 0)
    v_codec = str(video_stream.get("codec_name") or "").lower()
    a_codec = str(audio_stream.get("codec_name") or "").lower()

    duration_raw = video_stream.get("duration") or fmt.get("duration") or 0.0
    duration_sec = round(float(duration_raw), 3)

    fps_str = video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate") or "0/1"
    fps = 0.0
    if "/" in str(fps_str):
        num, den = str(fps_str).split("/", 1)
        if int(den) != 0:
            fps = round(float(Fraction(int(num), int(den))), 2)
    else:
        fps = round(float(fps_str), 2)

    if width != 1080 or height != 1920:
        raise RuntimeError(
            f"Resolução inválida em {video_path.name}: {width}x{height} (esperado 1080x1920)"
        )
    if v_codec != "h264":
        raise RuntimeError(
            f"Codec de vídeo inválido em {video_path.name}: {v_codec} (esperado h264)"
        )
    if a_codec != "aac":
        raise RuntimeError(
            f"Codec de áudio inválido em {video_path.name}: {a_codec} (esperado aac)"
        )
    if duration_sec <= 0:
        raise RuntimeError(f"Duração inválida em {video_path.name}: {duration_sec}s")

    return {
        "width": width,
        "height": height,
        "video_codec": v_codec,
        "audio_codec": a_codec,
        "fps": fps,
        "duration_sec": duration_sec,
        "size_bytes": video_path.stat().st_size,
    }


def fit_shorts_title(raw_title: str) -> str:
    """Garante título do YouTube Shorts com #Shorts dentro do limite de 100 caracteres."""
    base = " ".join((raw_title or "ASMR Satisfatório").split())
    suffix = " #Shorts"
    if "#shorts" in base.lower():
        return base[:100].strip()
    max_base = 100 - len(suffix)
    if len(base) > max_base:
        base = base[:max_base].rstrip(" -|:,.")
    return f"{base}{suffix}"


def build_platform_packages(
    record: dict[str, Any],
    video_path: Path,
    probe: dict[str, Any],
    handle_yt: str,
    handle_ig: str,
    handle_tt: str,
) -> list[dict[str, Any]]:
    """Monta os 3 pacotes (YouTube Shorts, Instagram Reels, TikTok) com cross-linking."""
    sc = record["shortcode"]
    categoria = record.get("categoria_fisica") or "satisfying_asmr"
    tema = (record.get("variacao_tema") or record.get("titulo_viral_pt") or sc).strip()
    titulo_pt = (record.get("titulo_viral_pt") or tema).strip()
    titulo_en = (record.get("titulo_viral_en") or "").strip()

    tags_list = parse_hashtags(record.get("hashtags"))
    tags_no_hash = [t.lstrip("#") for t in tags_list]
    if "Shorts" not in tags_no_hash:
        tags_no_hash.append("Shorts")

    hashtags_yt = " ".join(tags_list + (["#Shorts"] if "#shorts" not in {t.lower() for t in tags_list} else []))
    hashtags_ig = " ".join(tags_list[:10] + (["#Reels"] if "#reels" not in {t.lower() for t in tags_list} else []))
    hashtags_tt = " ".join(tags_list[:6] + (["#FYP"] if "#fyp" not in {t.lower() for t in tags_list} else []))

    video_rel = video_path.relative_to(BASE_DIR).as_posix()
    video_abs = str(video_path.resolve())

    # 1. YouTube Shorts
    yt_title = fit_shorts_title(titulo_pt)
    yt_desc_lines = [
        titulo_pt,
        f"{titulo_en}" if titulo_en else "",
        "",
        f"Som ASMR original masterizado em alta definição | Categoria: {categoria}",
        f"Transformação visual: {tema}",
        "",
        f"📲 Acompanhe também no Instagram ({handle_ig}) e TikTok ({handle_tt})",
        "",
        hashtags_yt,
    ]
    yt_desc = "\n".join(line for line in yt_desc_lines if line is not None).strip()
    yt_comment = (
        f"Qual material você quer ver nessa máquina no próximo vídeo? "
        f"Siga também {handle_ig} no Instagram e {handle_tt} no TikTok!"
    )

    # 2. Instagram Reels
    ig_caption = "\n".join(
        [
            f"{titulo_pt} 🎧✨",
            "",
            f"Aumente o som para o ASMR real ({probe['duration_sec']}s em 60fps).",
            f"📺 Veja a coleção completa no YouTube ({handle_yt}) e TikTok ({handle_tt}).",
            ".",
            ".",
            hashtags_ig,
        ]
    )
    ig_comment = (
        f"De 0 a 10, qual o nível de satisfação desse som? 👇 "
        f"Mais vídeos diários em {handle_yt} (YouTube) e {handle_tt} (TikTok)!"
    )

    # 3. TikTok
    tt_caption = (
        f"{titulo_pt} 🔥🎧 Som ASMR real sem cortes! "
        f"Siga também no IG ({handle_ig}) e YouTube ({handle_yt}) {hashtags_tt}"
    )
    tt_comment = (
        f"Ouça de fone 🎧🔥 Qual próximo objeto testamos? "
        f"IG: {handle_ig} | YT: {handle_yt}"
    )

    return [
        {
            "ref_shortcode": sc,
            "plataforma": "youtube_shorts",
            "video_final_path": video_rel,
            "titulo": yt_title,
            "legenda_completa": yt_desc,
            "hashtags": hashtags_yt,
            "primeiro_comentario": yt_comment,
            "status": "pronto",
            "post_url": None,
            "api_payload": {
                "endpoint": "YouTube Data API v3 - videos.insert",
                "file_path": video_abs,
                "snippet": {
                    "title": yt_title,
                    "description": yt_desc,
                    "tags": tags_no_hash,
                    "categoryId": "24",
                },
                "status": {"privacyStatus": "public", "selfDeclaredMadeForKids": False},
                "follow_up_comment": yt_comment,
            },
        },
        {
            "ref_shortcode": sc,
            "plataforma": "instagram_reels",
            "video_final_path": video_rel,
            "titulo": titulo_pt,
            "legenda_completa": ig_caption,
            "hashtags": hashtags_ig,
            "primeiro_comentario": ig_comment,
            "status": "pronto",
            "post_url": None,
            "api_payload": {
                "endpoint": "Instagram Graph API - POST /{ig-user-id}/media + media_publish",
                "file_path": video_abs,
                "media_type": "REELS",
                "share_to_feed": True,
                "caption": ig_caption,
                "follow_up_comment": ig_comment,
            },
        },
        {
            "ref_shortcode": sc,
            "plataforma": "tiktok",
            "video_final_path": video_rel,
            "titulo": titulo_pt,
            "legenda_completa": tt_caption,
            "hashtags": hashtags_tt,
            "primeiro_comentario": tt_comment,
            "status": "pronto",
            "post_url": None,
            "api_payload": {
                "endpoint": "TikTok Content Posting API - POST /v2/post/publish/video/init/",
                "file_path": video_abs,
                "post_info": {
                    "title": tt_caption[:2200],
                    "privacy_level": "PUBLIC_TO_EVERYONE",
                    "disable_duet": False,
                    "disable_stitch": False,
                },
                "source_info": {
                    "source": "FILE_UPLOAD",
                    "video_size": probe["size_bytes"],
                },
                "follow_up_comment": tt_comment,
            },
        },
    ]


def export_ready_folder(item: dict[str, Any]) -> Path:
    """Cria a pasta organizada output/prontos/<shortcode>/ com JSON e folha de postagem."""
    sc = item["shortcode"]
    ready_dir = PRONTOS_DIR / sc
    ready_dir.mkdir(parents=True, exist_ok=True)

    (ready_dir / "pacote_redes.json").write_text(
        json.dumps(item, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    p = item["video_specs"]
    lines: list[str] = [
        f"# Folha de Postagem — `{sc}`",
        "",
        f"- **Vídeo final:** `{(BASE_DIR / item['video_final_path']).resolve()}`",
        f"- **Especificações:** `{p['width']}x{p['height']}` | `{p['fps']} fps` | `{p['duration_sec']}s` | `{p['video_codec']}/{p['audio_codec']}` | `{round(p['size_bytes'] / (1024 * 1024), 2)} MB`",
        f"- **Tema criativo:** {item['variacao_tema']}",
        "",
        "---",
        "",
    ]
    for pub in item["plataformas"]:
        lines.extend(
            [
                f"## {pub['plataforma']}",
                f"- **Título:** {pub['titulo']}",
                f"- **Hashtags:** `{pub['hashtags']}`",
                f"- **Primeiro comentário:** {pub['primeiro_comentario']}",
                "",
                "**Legenda / Descrição:**",
                "```text",
                pub["legenda_completa"],
                "```",
                "",
            ]
        )

    (ready_dir / "FOLHA_POSTAGEM.md").write_text("\n".join(lines), encoding="utf-8")
    return ready_dir


def export_publication_package(
    bundle_items: list[dict[str, Any]],
    out_prefix: str = DEFAULT_OUT_PREFIX,
) -> tuple[Path, Path]:
    """Exporta JSON, Markdown e pastas individuais output/prontos/<shortcode>/."""
    ensure_publishing_structure()
    for item in bundle_items:
        export_ready_folder(item)
    json_path = OUTPUT_DIR / f"{out_prefix}.json"
    md_path = OUTPUT_DIR / f"{out_prefix}.md"

    json_path.write_text(
        json.dumps(bundle_items, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    md_lines: list[str] = [
        "# Pacote de Publicação Multi-Rede (Esteira Ponta a Ponta)",
        "",
        f"Total de vídeos prontos: **{len(bundle_items)}** | Total de publicações preparadas: **{sum(len(i['plataformas']) for i in bundle_items)}**",
        "",
        "---",
        "",
    ]

    for idx, item in enumerate(bundle_items, start=1):
        p = item["video_specs"]
        md_lines.extend(
            [
                f"## {idx}. `{item['shortcode']}` — {item['categoria_fisica']}",
                f"- **Arquivo final:** `{item['video_final_path']}`",
                f"- **Especificações FFprobe:** `{p['width']}x{p['height']}` | `{p['fps']} fps` | `{p['duration_sec']}s` | vídeo `{p['video_codec']}` | áudio `{p['audio_codec']}` | `{round(p['size_bytes'] / (1024 * 1024), 2)} MB`",
                f"- **Tema criativo (Llama 3.1):** {item['variacao_tema']}",
                "",
            ]
        )
        for pub in item["plataformas"]:
            md_lines.extend(
                [
                    f"### Plataforma: `{pub['plataforma']}` (`status={pub['status']}`)",
                    f"- **Título:** {pub['titulo']}",
                    f"- **Hashtags:** `{pub['hashtags']}`",
                    f"- **Primeiro comentário automático:** {pub['primeiro_comentario']}",
                    "",
                    "**Legenda / Descrição completa:**",
                    "```text",
                    pub["legenda_completa"],
                    "```",
                    "",
                ]
            )
        md_lines.extend(["---", ""])

    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    return json_path, md_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Valida os vídeos finais e gera pacotes de publicação cruzada (YouTube, Instagram, TikTok)."
    )
    parser.add_argument(
        "--shortcodes",
        nargs="+",
        required=True,
        help="Lista de shortcodes a processar na etapa final de publicação.",
    )
    parser.add_argument(
        "--handle-yt",
        type=str,
        default=DEFAULT_HANDLE,
        help=f"Handle do canal no YouTube (padrao: {DEFAULT_HANDLE}).",
    )
    parser.add_argument(
        "--handle-ig",
        type=str,
        default=DEFAULT_HANDLE,
        help=f"Handle da conta no Instagram (padrao: {DEFAULT_HANDLE}).",
    )
    parser.add_argument(
        "--handle-tt",
        type=str,
        default=DEFAULT_HANDLE,
        help=f"Handle da conta no TikTok (padrao: {DEFAULT_HANDLE}).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["prepare"],
        default="prepare",
        help="Modo de operacao (padrao: prepare).",
    )
    parser.add_argument(
        "--out-prefix",
        type=str,
        default=DEFAULT_OUT_PREFIX,
        help=f"Prefixo dos arquivos exportados em output/ (padrao: {DEFAULT_OUT_PREFIX}).",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Limpa registros anteriores da tabela publicacoes antes de gravar o lote atual.",
    )
    return parser.parse_args()


def main() -> int:
    db.init_db()
    ensure_publishing_structure()
    args = parse_args()

    if args.reset:
        with db.get_connection() as conn:
            conn.execute("DELETE FROM publicacoes")
            conn.commit()

    handle_yt = normalize_handle(args.handle_yt)
    handle_ig = normalize_handle(args.handle_ig)
    handle_tt = normalize_handle(args.handle_tt)

    catalog = {row["shortcode"]: row for row in db.get_full_catalog()}
    bundle_items: list[dict[str, Any]] = []
    errors = 0

    for sc in args.shortcodes:
        record = catalog.get(sc)
        if not record:
            print(f"[ERRO] Shortcode '{sc}' nao encontrado na base de conhecimento.", file=sys.stderr)
            errors += 1
            continue

        video_path = resolve_final_video(sc)
        if not video_path:
            print(
                f"[ERRO] Nenhum MP4 masterizado encontrado para '{sc}' em output/.",
                file=sys.stderr,
            )
            errors += 1
            continue

        try:
            probe = probe_final_video(video_path)
        except Exception as exc:
            print(f"[ERRO] Falha na validacao de '{sc}': {exc}", file=sys.stderr)
            errors += 1
            continue

        platforms = build_platform_packages(
            record=record,
            video_path=video_path,
            probe=probe,
            handle_yt=handle_yt,
            handle_ig=handle_ig,
            handle_tt=handle_tt,
        )

        pub_ids: list[int] = []
        for pub in platforms:
            pub_id = db.upsert_publicacao(pub)
            pub_ids.append(pub_id)

        print(
            f"[OK] {sc} -> {video_path.name} ({probe['width']}x{probe['height']} @ {probe['fps']}fps, "
            f"{probe['duration_sec']}s, {probe['video_codec']}/{probe['audio_codec']}) | "
            f"3 redes salvas (ids={pub_ids})"
        )

        bundle_items.append(
            {
                "shortcode": sc,
                "categoria_fisica": record.get("categoria_fisica"),
                "variacao_tema": record.get("variacao_tema"),
                "video_final_path": video_path.relative_to(BASE_DIR).as_posix(),
                "video_specs": probe,
                "handles": {
                    "youtube_shorts": handle_yt,
                    "instagram_reels": handle_ig,
                    "tiktok": handle_tt,
                },
                "plataformas": platforms,
            }
        )

    if not bundle_items:
        return 1

    json_path, md_path = export_publication_package(bundle_items, out_prefix=args.out_prefix)
    print(f"\nPacotes de publicacao exportados ({args.mode}):")
    print(f"  - {json_path.relative_to(BASE_DIR).as_posix()}")
    print(f"  - {md_path.relative_to(BASE_DIR).as_posix()}")

    return 1 if errors > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
