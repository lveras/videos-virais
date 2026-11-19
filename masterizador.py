"""Masterizador final FFmpeg (mux com vídeo Wan 2.1 + áudio ASMR e preview remix 9:16)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import db

BASE_DIR = Path(__file__).resolve().parent
BAIXADOS_DIR = BASE_DIR / "baixados"
CRIADOS_DIR = BASE_DIR / "criados"
OUTPUT_DIR = BASE_DIR / "output"


def run_ffmpeg(cmd: list[str]) -> None:
    """Executa FFmpeg e lança erro legível caso falhe."""
    res = subprocess.run(
        cmd,
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if res.returncode != 0:
        raise RuntimeError(f"FFmpeg falhou (code={res.returncode}):\n{res.stderr.strip()}")


def resolve_reference_paths(shortcode: str) -> tuple[Path, Path | None]:
    """Resolve vídeo de referência e áudio ASMR a partir do SQLite ou caminhos padrão."""
    record = db.get_video_by_shortcode(shortcode)
    fallback_video = BAIXADOS_DIR / f"{shortcode}.mp4"
    if record:
        candidate_video = BASE_DIR / record["video_path"]
        video_path = candidate_video if candidate_video.exists() else fallback_video
        audio_path = (
            BASE_DIR / record["audio_asmr_path"]
            if record.get("audio_asmr_path")
            else None
        )
    else:
        video_path = fallback_video
        candidate_audio = BASE_DIR / "referencias" / "audios_asmr" / f"{shortcode}_asmr.wav"
        audio_path = candidate_audio if candidate_audio.exists() else None

    return video_path, audio_path


def mux_ai_video(shortcode: str, ai_video_path: Path) -> Path:
    """Combina o vídeo renderizado pelo Wan 2.1 com o áudio ASMR em 1080x1920."""
    if not ai_video_path.exists():
        raise FileNotFoundError(f"Video de IA nao encontrado: {ai_video_path}")

    _, audio_asmr_path = resolve_reference_paths(shortcode)
    CRIADOS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CRIADOS_DIR / f"{shortcode}_final_viral.mp4"

    vf = (
        "scale=1080:1920:force_original_aspect_ratio=decrease,"
        "pad=1080:1920:(ow-iw)/2:(oh-ih)/2,"
        "unsharp=5:5:0.6:5:5:0.0,format=yuv420p"
    )

    if audio_asmr_path and audio_asmr_path.exists():
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(ai_video_path),
            "-i",
            str(audio_asmr_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "17",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            "-movflags",
            "+faststart",
            str(out_path),
        ]
    else:
        print(
            f"[AVISO] Audio ASMR para {shortcode} nao encontrado; gerando video sem faixa de audio."
        )
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(ai_video_path),
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "17",
            "-an",
            "-movflags",
            "+faststart",
            str(out_path),
        ]

    run_ffmpeg(cmd)
    return out_path


def preview_remix(shortcode: str) -> Path:
    """Gera preview imediato em 1080x1920 60fps com grade macro industrial + áudio ASMR."""
    ref_video_path, audio_asmr_path = resolve_reference_paths(shortcode)
    if not ref_video_path.exists():
        raise FileNotFoundError(
            f"Video base nao encontrado para {shortcode}: {ref_video_path}"
        )

    CRIADOS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CRIADOS_DIR / f"{shortcode}_preview_asmr_remix.mp4"

    vf = (
        "scale=1080:1920:force_original_aspect_ratio=decrease,"
        "pad=1080:1920:(ow-iw)/2:(oh-ih)/2,"
        "eq=contrast=1.12:saturation=1.22:brightness=0.01,"
        "unsharp=5:5:0.8:5:5:0.0,"
        "fps=60,format=yuv420p"
    )

    if audio_asmr_path and audio_asmr_path.exists():
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(ref_video_path),
            "-i",
            str(audio_asmr_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            "-movflags",
            "+faststart",
            str(out_path),
        ]
    else:
        print(
            f"[AVISO] Audio ASMR para {shortcode} nao encontrado; gerando preview apenas com video."
        )
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(ref_video_path),
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-an",
            "-movflags",
            "+faststart",
            str(out_path),
        ]

    run_ffmpeg(cmd)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Masterizador FFmpeg para finalizar videos Wan 2.1 ou gerar previews ASMR."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--mux",
        action="store_true",
        help="Une um video gerado no ComfyUI/Wan 2.1 com o audio ASMR masterizado.",
    )
    mode.add_argument(
        "--preview-remix",
        action="store_true",
        help="Cria um preview imediato com o audio ASMR masterizado e grade visual macro.",
    )
    parser.add_argument(
        "--shortcode",
        type=str,
        required=True,
        help="Shortcode/ID do video na base de conhecimento.",
    )
    parser.add_argument(
        "--ai-video",
        type=Path,
        default=None,
        help="Caminho do MP4 gerado pelo Wan 2.1 (obrigatorio com --mux).",
    )
    return parser.parse_args()


def main() -> int:
    db.init_db()
    args = parse_args()

    try:
        if args.mux:
            if not args.ai_video:
                print("[ERRO] --ai-video e obrigatorio quando --mux e usado.", file=sys.stderr)
                return 1
            out = mux_ai_video(args.shortcode, args.ai_video)
            print(f"Video final masterizado: {out}")
        elif args.preview_remix:
            out = preview_remix(args.shortcode)
            print(f"Preview ASMR remix gerado: {out}")
    except Exception as exc:
        print(f"[ERRO] {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
