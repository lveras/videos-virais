"""Diretor Criativo local usando Ollama (llama3.1:latest) para Wan 2.1 Video-to-Video."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import db

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "llama3.1:latest"


def check_ollama_alive() -> list[str] | None:
    """Verifica se o servidor local do Ollama está respondendo e lista os modelos."""
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            return [m.get("name", "") for m in payload.get("models", [])]
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return None


def ensure_ollama_running(model: str = DEFAULT_MODEL) -> None:
    """Inicia `ollama serve` com aceleração Vulkan (RX 9070 XT) se a API não estiver ativa."""
    os.environ["OLLAMA_VULKAN"] = "1"
    models = check_ollama_alive()
    if models is None:
        print("Ollama nao esta rodando. Iniciando `ollama serve` com OLLAMA_VULKAN=1...")
        env = os.environ.copy()
        env["OLLAMA_VULKAN"] = "1"
        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
        try:
            subprocess.Popen(
                ["ollama", "serve"],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                creationflags=creationflags,
            )
        except OSError as exc:
            raise RuntimeError(f"Nao foi possivel iniciar `ollama serve`: {exc}") from exc

        deadline = time.monotonic() + 45.0
        while time.monotonic() < deadline:
            time.sleep(1.5)
            models = check_ollama_alive()
            if models is not None:
                break

        if models is None:
            raise RuntimeError("Timeout de 45s aguardando `ollama serve` responder.")

    print(f"Ollama ativo. Modelos disponiveis: {', '.join(models) if models else 'nenhum'}")
    if models and model not in models and f"{model}:latest" not in models:
        print(f"[AVISO] Modelo '{model}' nao listado exatamente em /api/tags; tentando usar assim mesmo.")


def build_director_prompt(video: dict[str, Any]) -> str:
    """Monta o prompt técnico para o Llama 3.1 gerar a variação Wan 2.1 Vid2Vid."""
    categoria = (video.get("categoria_fisica") or "").lower()
    style_hint = ""
    if "rug_cleaning" in categoria or "carpet" in categoria:
        style_hint = """
DIRETRIZ ESPECÍFICA ESTILO @MagnusRugCleaning (Limpeza de Carpete ASMR):
- Preserve EXATAMENTE o movimento do rodo raspador metálico (metal squeegee/scraper), escova rotativa e bocal extrator transparente.
- Transforme a lama, espuma e trama do tapete em combinações ultra-hipnóticas de alto contraste: ex. espuma neon turquesa, magenta ou dourada sendo raspada em linhas retas perfeitas de um tapete persa luxuoso ou tapete geométrico branco puro, água densa escorrendo e fibras macro detalhadas."""
    elif "hydraulic_press" in categoria:
        style_hint = """
DIRETRIZ ESPECÍFICA ESTILO @HydraulicPressChannel (Prensa Hidráulica):
- Preserve EXATAMENTE a descida vertical do pistão cilíndrico de aço pesado e a física real de compressão, esmagamento ou estilhaçamento.
- Transforme o objeto esmagado e a iluminação em materiais de altíssimo apelo visual: ex. xícaras de cristal de quartzo ou porcelana dourada estilhaçando em câmera lenta, discos de lava/borracha neon extrudando sob toneladas de pressão, iluminação industrial dramática com fragmentos e partículas nítidas."""

    return f"""Você é um Diretor Técnico de Vídeos Virais ASMR / Oddly Satisfying e Engenheiro de Prompts especialista em Wan 2.1 Video-to-Video (ComfyUI).

Analise o vídeo de referência real abaixo da nossa Base de Conhecimento:
- Shortcode: {video.get('shortcode')}
- Categoria física/mecânica: {video.get('categoria_fisica')}
- Criador original: {video.get('uploader') or 'Desconhecido'}
- Duração: {video.get('duration_sec')}s | Resolução: {video.get('width')}x{video.get('height')} @ {video.get('fps')}fps
- Descrição visual e física da ação real: {video.get('descricao_visual')}
- Legenda original: {video.get('caption') or ''}
{style_hint}

MISSÃO TÉCNICA OBRIGATÓRIA:
No pipeline Wan 2.1 Video-to-Video, o vídeo real será usado como guia estrutural e seu áudio ASMR real será mantido 100% sincronizado.
Portanto, você DEVE preservar EXATAMENTE:
1. A física do movimento, velocidade, direção e ângulo de câmera (vertical 9:16 macro close-up).
2. A silhueta e geometria das ferramentas/máquinas/mãos para que o som ASMR mecânico original continue combinando perfeitamente.
3. Por cima dessa física idêntica, proponha uma RE-ESTILIZAÇÃO VISUAL HIPNOTIZANTE DE ALTO CONTRASTE (materiais premium, texturas luxuosas, cores vibrantes ou contrastes extremos).

Responda EXCLUSIVAMENTE em JSON válido com exatamente estas 8 chaves (substituindo todos os placeholders `<...>` por conteúdo específico para ESTE vídeo):
{{
  "variacao_tema": "<1 frase concreta em PT-BR nomeando o material original e o novo material visual escolhido para este vídeo>",
  "wan21_positive_prompt": "<prompt concreto de 40 a 65 palavras EM INGLÊS descrevendo exatamente: enquadramento vertical 9:16 macro, ferramenta/máquina da cena, movimento físico exato, materiais novos específicos (ex: 24k gold, obsidian, neon turquoise foam, crystal glass, polished steel), iluminação e reflexos 8k UHD>",
  "wan21_negative_prompt": "blur, morphing, distorted geometry, extra hands, warped tools, melting metal artifacts, watermark, text overlay, subtitles, low resolution, flicker, jitter",
  "recommended_denoise": 0.55,
  "titulo_viral_pt": "<título curto e magnético em Português específico para este objeto/ação, máx 65 chars>",
  "titulo_viral_en": "<short high-retention English title specific to this object/action, max 65 chars>",
  "hashtags": "<5 a 7 hashtags específicas do nicho em inglês/português>",
  "notas_fisica_som": "<1 frase em PT-BR explicando como o denoise escolhido e a nova textura mantêm a sincronia com o som ASMR>"
}}

REGRAS ADICIONAIS:
- PROIBIDO copiar textos genéricos como "materiais premium", "texturas luxuosas", "Detailed technical English prompt" ou "exact physical motion preservation". Nomeie os materiais e cores concretos!
- `recommended_denoise` DEVE ser um número decimal entre 0.48 e 0.62 (use valores mais baixos como 0.48-0.52 para ações rápidas/precisas com mãos ou cavacos finos, e 0.54-0.62 para superfícies grandes/prensas/rolos).
- `wan21_positive_prompt` e `wan21_negative_prompt` DEVEM estar em inglês técnico e ricos em detalhes de textura, iluminação e câmera.
- Não inclua texto fora do JSON."""


def call_llama_json(prompt: str, model: str = DEFAULT_MODEL) -> dict[str, Any]:
    """Chama a API local do Ollama com saída forçada em JSON."""
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {
                "temperature": 0.65,
                "num_ctx": 4096,
            },
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        raw_resp = json.loads(resp.read().decode("utf-8"))

    content = (raw_resp.get("response") or "").strip()
    if content.startswith("```"):
        lines = content.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines).strip()

    parsed = json.loads(content)
    return parsed


def normalize_roteiro(shortcode: str, raw: dict[str, Any]) -> dict[str, Any]:
    """Valida e normaliza os campos retornados pelo Llama 3.1."""
    try:
        denoise = float(raw.get("recommended_denoise", 0.55))
    except (TypeError, ValueError):
        denoise = 0.55
    denoise = round(max(0.48, min(0.62, denoise)), 2)

    hashtags = raw.get("hashtags") or "#asmr #oddlysatisfying #satisfying"
    if isinstance(hashtags, list):
        hashtags = " ".join(str(t) for t in hashtags)

    neg_default = (
        "blur, morphing, distorted tools, watermark, text, subtitles, "
        "low quality, flicker, deformed geometry"
    )
    return {
        "ref_shortcode": shortcode,
        "variacao_tema": str(raw.get("variacao_tema") or "").strip(),
        "wan21_positive_prompt": str(raw.get("wan21_positive_prompt") or "").strip(),
        "wan21_negative_prompt": str(
            raw.get("wan21_negative_prompt") or neg_default
        ).strip(),
        "recommended_denoise": denoise,
        "titulo_viral_pt": str(raw.get("titulo_viral_pt") or "").strip(),
        "titulo_viral_en": str(raw.get("titulo_viral_en") or "").strip(),
        "hashtags": str(hashtags).strip(),
        "notas_fisica_som": str(raw.get("notas_fisica_som") or "").strip(),
    }


def export_production_outputs() -> tuple[Path, Path]:
    """Exporta `output/fila_wan21_jobs.json` e `output/catalogo_producao.md`."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    catalog = db.get_full_catalog()

    jobs: list[dict[str, Any]] = []
    md_lines: list[str] = [
        "# Catálogo de Produção Viral (Base de Conhecimento + Llama 3.1 + Wan 2.1)",
        "",
        f"Total de vídeos catalogados: **{len(catalog)}**",
        "",
        "---",
        "",
    ]

    for idx, row in enumerate(catalog, start=1):
        sc = row["shortcode"]
        video_abs = str((BASE_DIR / row["video_path"]).resolve()) if row.get("video_path") else ""
        audio_abs = (
            str((BASE_DIR / row["audio_asmr_path"]).resolve())
            if row.get("audio_asmr_path")
            else ""
        )
        frames = []
        if row.get("frames_json"):
            try:
                frames = json.loads(row["frames_json"])
            except json.JSONDecodeError:
                frames = []

        jobs.append(
            {
                "job_index": idx,
                "shortcode": sc,
                "categoria_fisica": row.get("categoria_fisica"),
                "guide_video_path": video_abs,
                "audio_asmr_path": audio_abs,
                "duration_sec": row.get("duration_sec"),
                "source_width": row.get("width"),
                "source_height": row.get("height"),
                "source_fps": row.get("fps"),
                "target_wan21_resolution": {"width": 480, "height": 832},
                "denoise": row.get("recommended_denoise") or 0.55,
                "positive_prompt": row.get("wan21_positive_prompt") or "",
                "negative_prompt": row.get("wan21_negative_prompt") or "",
                "variacao_tema": row.get("variacao_tema") or "",
                "titulo_viral_pt": row.get("titulo_viral_pt") or "",
                "titulo_viral_en": row.get("titulo_viral_en") or "",
                "hashtags": row.get("hashtags") or "",
            }
        )

        md_lines.extend(
            [
                f"## {idx}. `{sc}` — {row.get('categoria_fisica')}",
                f"- **Origem:** {row.get('url')} (`{row.get('uploader') or 'N/A'}`)",
                f"- **Métricas originais:** likes={row.get('likes') if row.get('likes') is not None else 'N/A'} | comentários={row.get('comments') if row.get('comments') is not None else 'N/A'}",
                f"- **Especificações:** `{row.get('width')}x{row.get('height')}` @ `{row.get('fps')} fps` | `{row.get('duration_sec')}s`",
                f"- **Vídeo guia:** `{row.get('video_path')}`",
                f"- **Áudio ASMR masterizado:** `{row.get('audio_asmr_path') or 'N/A'}`",
                f"- **Frames-chave:** {', '.join(f'`{f}`' for f in frames) if frames else 'N/A'}",
                "",
                "### Direção Criativa (Llama 3.1)",
                f"- **Variação visual:** {row.get('variacao_tema') or 'Pendente'}",
                f"- **Denoise recomendado:** `{row.get('recommended_denoise')}`",
                f"- **Título Viral (PT):** {row.get('titulo_viral_pt') or 'N/A'}",
                f"- **Título Viral (EN):** {row.get('titulo_viral_en') or 'N/A'}",
                f"- **Hashtags:** `{row.get('hashtags') or ''}`",
                f"- **Notas Física & Som ASMR:** {row.get('notas_fisica_som') or 'N/A'}",
                "",
                "**Prompt Positivo (Wan 2.1):**",
                "```text",
                row.get("wan21_positive_prompt") or "",
                "```",
                "",
                "**Prompt Negativo (Wan 2.1):**",
                "```text",
                row.get("wan21_negative_prompt") or "",
                "```",
                "",
                "---",
                "",
            ]
        )

    jobs_path = OUTPUT_DIR / "fila_wan21_jobs.json"
    md_path = OUTPUT_DIR / "catalogo_producao.md"
    jobs_path.write_text(
        json.dumps(jobs, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    return jobs_path, md_path


def unload_ollama_model(model: str = DEFAULT_MODEL) -> None:
    """Descarrega o modelo Ollama da VRAM (keep_alive=0) para liberar a RX 9070 XT."""
    try:
        body = json.dumps({"model": model, "keep_alive": 0}).encode("utf-8")
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read()
        print(f"VRAM liberada: modelo '{model}' descarregado do Ollama (keep_alive=0).")
    except (urllib.error.URLError, TimeoutError, OSError):
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diretor Criativo local usando Ollama Llama 3.1 para gerar prompts Wan 2.1."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Nome do modelo Ollama (padrao: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenera roteiros/prompts mesmo para videos que ja possuem roteiro no SQLite.",
    )
    return parser.parse_args()


def main() -> int:
    db.init_db()
    args = parse_args()

    try:
        ensure_ollama_running(args.model)
    except RuntimeError as exc:
        print(f"[ERRO] {exc}", file=sys.stderr)
        return 1

    videos = db.get_all_videos() if args.force else db.get_videos_without_roteiro()
    if not videos:
        print("Todos os videos da base ja possuem roteiros gerados. Atualizando arquivos de saida...")
    else:
        print(f"Gerando direcao criativa com {args.model} para {len(videos)} video(s)...")

    errors = 0
    for video in videos:
        sc = video["shortcode"]
        print(f"\n==> Llama 3.1 analisando [{sc}] ({video.get('categoria_fisica')})...")
        prompt = build_director_prompt(video)
        try:
            raw_json = call_llama_json(prompt, model=args.model)
            roteiro = normalize_roteiro(sc, raw_json)
            row_id = db.insert_roteiro_ia(roteiro, replace_existing=args.force)
            print(f"    OK (roteiro id={row_id} | denoise={roteiro['recommended_denoise']})")
            print(f"    Tema: {roteiro['variacao_tema']}")
            print(f"    Titulo PT: {roteiro['titulo_viral_pt']}")
        except Exception as exc:
            errors += 1
            print(f"    [ERRO] Falha ao gerar roteiro para {sc}: {exc}", file=sys.stderr)

    jobs_path, md_path = export_production_outputs()
    print(f"\nArquivos exportados:")
    print(f"  - {jobs_path.relative_to(BASE_DIR).as_posix()}")
    print(f"  - {md_path.relative_to(BASE_DIR).as_posix()}")

    if videos:
        unload_ollama_model(args.model)

    if errors > 0 and errors == len(videos):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
