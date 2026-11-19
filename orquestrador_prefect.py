"""Orquestrador local com Prefect para a linha de montagem de vídeos virais."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from prefect import flow, get_run_logger, runtime, task

import db

BASE_DIR = Path(__file__).resolve().parent
REFERENCIAS_VIDEOS_DIR = BASE_DIR / "referencias" / "videos"
BAIXADOS_DIR = BASE_DIR / "baixados"
CRIADOS_DIR = BASE_DIR / "criados"
OUTPUT_DIR = BASE_DIR / "output"

ROCM_PYTHON = Path(
    r"C:\Users\lucia\AppData\Local\AMD\AI_Bundle\ComfyUI\venv\Scripts\python.exe"
)

TARGET_SHORTCODES = ["FDa1Oa2y0NY", "DfNRXxjou-k", "Df6LIe7uxxw", "uhQXBMZ-sTg"]

ORIGINAL_SCENE_FILES: dict[str, str] = {
    "cristal": "criado_01_prensa_cristal_esmeralda.mp4",
    "capsula": "criado_02_corte_capsula_iridescente.mp4",
    "torno": "criado_03_torno_helicoidal_damasco.mp4",
    "rodo": "criado_04_rodo_espuma_prismatica.mp4",
}

FLOW_NAME = "linha_de_montagem_viral"


def _current_run_id() -> str:
    try:
        rid = runtime.flow_run.id
        if rid:
            return str(rid)
    except Exception:
        pass
    return str(uuid.uuid4())


def _run_module(cmd: list[str], stage_name: str) -> str:
    """Executa um módulo independente como subprocesso mantendo isolamento de memória/VRAM."""
    res = subprocess.run(
        cmd,
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    output = (res.stdout or "").strip()
    err = (res.stderr or "").strip()
    if res.returncode != 0:
        raise RuntimeError(
            f"Etapa '{stage_name}' falhou (code={res.returncode}).\nSTDOUT:\n{output}\nSTDERR:\n{err}"
        )
    return output


def _is_gpu_generator_running() -> bool:
    """Verifica no Windows se já existe um processo executando gerador_original_gpu.py."""
    ps_cmd = (
        "Get-CimInstance Win32_Process -Filter \"Name LIKE 'python%'\" | "
        "Where-Object { $_.CommandLine -like '*gerador_original_gpu.py*' } | "
        "Select-Object -ExpandProperty ProcessId"
    )
    res = subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps_cmd],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return bool(res.stdout.strip())


@task(name="01_verificar_modelos", retries=2, retry_delay_seconds=5)
def task_01_verificar_modelos(run_id: str) -> list[dict[str, Any]]:
    """Verifica modelos locais, integra caminhos do ComfyUI e atualiza tabela modelos_locais."""
    logger = get_run_logger()
    db.log_pipeline_step(run_id, FLOW_NAME, "01_verificar_modelos", "running")
    try:
        out = _run_module(
            [sys.executable, "gerenciador_modelos.py", "--action", "check"],
            "01_verificar_modelos",
        )
        logger.info(out)
        modelos = db.get_modelos_locais()
        faltantes = [m["model_key"] for m in modelos if m["obrigatorio"] and m["status"] != "instalado"]
        if faltantes:
            raise RuntimeError(f"Modelos/runtimes obrigatorios ausentes: {', '.join(faltantes)}")
        db.log_pipeline_step(
            run_id,
            FLOW_NAME,
            "01_verificar_modelos",
            "completed",
            detalhes=f"{len(modelos)} modelos/runtimes verificados no SQLite.",
        )
        return modelos
    except Exception as exc:
        db.log_pipeline_step(run_id, FLOW_NAME, "01_verificar_modelos", "failed", detalhes=str(exc))
        raise


@task(name="02_minerar_referencias", retries=2, retry_delay_seconds=5)
def task_02_minerar_referencias(run_id: str) -> dict[str, int]:
    """Executa minerador.py e sincroniza referências de vídeo com baixados/."""
    logger = get_run_logger()
    db.log_pipeline_step(run_id, FLOW_NAME, "02_minerar_referencias", "running")
    try:
        out = _run_module([sys.executable, "minerador.py"], "02_minerar_referencias")
        logger.info(out)

        BAIXADOS_DIR.mkdir(parents=True, exist_ok=True)
        copied = 0
        if REFERENCIAS_VIDEOS_DIR.exists():
            for src in REFERENCIAS_VIDEOS_DIR.glob("*.mp4"):
                dst = BAIXADOS_DIR / src.name
                if not dst.exists() or dst.stat().st_size == 0:
                    shutil.copy2(src, dst)
                    copied += 1

        total = len(db.get_all_videos())
        db.log_pipeline_step(
            run_id,
            FLOW_NAME,
            "02_minerar_referencias",
            "completed",
            detalhes=f"{total} videos indexados; {copied} sincronizados em baixados/.",
        )
        return {"videos_indexados": total, "sincronizados_baixados": copied}
    except Exception as exc:
        db.log_pipeline_step(run_id, FLOW_NAME, "02_minerar_referencias", "failed", detalhes=str(exc))
        raise


@task(name="03_diretor_criativo_llama", retries=1, retry_delay_seconds=5)
def task_03_diretor_llama(run_id: str) -> dict[str, int]:
    """Gera roteiros/prompts com Llama 3.1 (Vulkan) e descarrega a VRAM ao final."""
    logger = get_run_logger()
    db.log_pipeline_step(run_id, FLOW_NAME, "03_diretor_criativo_llama", "running")
    try:
        out = _run_module([sys.executable, "diretor_llama.py"], "03_diretor_criativo_llama")
        logger.info(out)
        catalog = db.get_full_catalog()
        prontos = sum(1 for r in catalog if r.get("roteiro_id"))
        db.log_pipeline_step(
            run_id,
            FLOW_NAME,
            "03_diretor_criativo_llama",
            "completed",
            detalhes=f"{prontos}/{len(catalog)} videos com roteiro Llama 3.1 pronto e VRAM liberada.",
        )
        return {"roteiros_prontos": prontos}
    except Exception as exc:
        db.log_pipeline_step(run_id, FLOW_NAME, "03_diretor_criativo_llama", "failed", detalhes=str(exc))
        raise


@task(name="04_renderizar_gpu_original", retries=1, retry_delay_seconds=10)
def task_04_renderizar_videos_gpu(run_id: str, force_render: bool = False) -> list[str]:
    """Aguarda render GPU ativo (se houver) e renderiza apenas cenas faltantes em criados/."""
    logger = get_run_logger()
    db.log_pipeline_step(run_id, FLOW_NAME, "04_renderizar_gpu_original", "running")
    try:
        CRIADOS_DIR.mkdir(parents=True, exist_ok=True)
        waited_sec = 0
        while _is_gpu_generator_running():
            if waited_sec == 0:
                logger.info("Detectado processo ativo de gerador_original_gpu.py na RX 9070 XT. Aguardando conclusao...")
            time.sleep(5)
            waited_sec += 5

        if force_render:
            missing_keys = list(ORIGINAL_SCENE_FILES.keys())
        else:
            missing_keys = [
                key
                for key, fname in ORIGINAL_SCENE_FILES.items()
                if not (CRIADOS_DIR / fname).exists() or (CRIADOS_DIR / fname).stat().st_size < 100_000
            ]

        if missing_keys:
            if not ROCM_PYTHON.exists():
                raise RuntimeError(f"Interpretador ROCm nao encontrado: {ROCM_PYTHON}")
            logger.info(f"Renderizando cenas pendentes na RX 9070 XT: {', '.join(missing_keys)}")
            out = _run_module(
                [str(ROCM_PYTHON), "gerador_original_gpu.py", "--scenes", *missing_keys],
                "04_renderizar_gpu_original",
            )
            logger.info(out)
        else:
            logger.info("Todos os 4 videos originais em criados/ ja estao renderizados.")

        rendered = [
            fname
            for fname in ORIGINAL_SCENE_FILES.values()
            if (CRIADOS_DIR / fname).exists() and (CRIADOS_DIR / fname).stat().st_size > 0
        ]
        db.log_pipeline_step(
            run_id,
            FLOW_NAME,
            "04_renderizar_gpu_original",
            "completed",
            detalhes=f"{len(rendered)}/4 videos em criados/ prontos (aguardou={waited_sec}s; renderizou={missing_keys}).",
        )
        return rendered
    except Exception as exc:
        db.log_pipeline_step(run_id, FLOW_NAME, "04_renderizar_gpu_original", "failed", detalhes=str(exc))
        raise


@task(name="05_masterizar_asmr", retries=1, retry_delay_seconds=5)
def task_05_masterizar_asmr(run_id: str) -> list[str]:
    """Garante previews remixados ASMR em output/ para os shortcodes principais."""
    logger = get_run_logger()
    db.log_pipeline_step(run_id, FLOW_NAME, "05_masterizar_asmr", "running")
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        generated: list[str] = []
        for sc in TARGET_SHORTCODES:
            preview_path = OUTPUT_DIR / f"{sc}_preview_asmr_remix.mp4"
            if not preview_path.exists() or preview_path.stat().st_size == 0:
                logger.info(f"Gerando preview ASMR remix para {sc}...")
                out = _run_module(
                    [sys.executable, "masterizador.py", "--preview-remix", "--shortcode", sc],
                    f"05_masterizar_asmr:{sc}",
                )
                logger.info(out)
            generated.append(preview_path.name)

        db.log_pipeline_step(
            run_id,
            FLOW_NAME,
            "05_masterizar_asmr",
            "completed",
            detalhes=f"Previews ASMR validados: {', '.join(generated)}",
        )
        return generated
    except Exception as exc:
        db.log_pipeline_step(run_id, FLOW_NAME, "05_masterizar_asmr", "failed", detalhes=str(exc))
        raise


def export_orchestrator_report(run_id: str) -> Path:
    """Gera relatório consolidado da linha de montagem Prefect a partir do SQLite."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    modelos = db.get_modelos_locais()
    videos = db.get_all_videos()
    catalog = db.get_full_catalog()
    pubs = db.get_publicacoes(TARGET_SHORTCODES)
    execs = [e for e in db.get_pipeline_execucoes(limit=30) if e["run_id"] == run_id]
    execs.reverse()

    lines: list[str] = [
        "# Relatório da Linha de Montagem Local (Prefect + SQLite)",
        "",
        f"- **Flow:** `{FLOW_NAME}`",
        f"- **Run ID:** `{run_id}`",
        f"- **Modelos/Runtimes registrados:** {len(modelos)} ({sum(1 for m in modelos if m['status'] == 'instalado')} instalados)",
        f"- **Vídeos de referência indexados:** {len(videos)}",
        f"- **Roteiros Llama 3.1 prontos:** {sum(1 for r in catalog if r.get('roteiro_id'))}",
        f"- **Publicações empacotadas (`output/prontos/`):** {len(pubs)}",
        "",
        "## Execução das Etapas neste Run",
        "",
        "| Etapa | Status | Detalhes |",
        "| :--- | :--- | :--- |",
    ]
    for e in execs:
        det = (e.get("detalhes") or "").replace("\n", " ").replace("|", "/")
        lines.append(f"| `{e['etapa']}` | `{e['status']}` | {det} |")

    lines.extend(
        [
            "",
            "## Inventário de Modelos (`modelos_locais`)",
            "",
            "| Chave | Provider | Status | Tamanho (MB) | Caminho |",
            "| :--- | :--- | :--- | ---: | :--- |",
        ]
    )
    for m in modelos:
        sz = f"{m['size_mb']:.1f}" if m.get("size_mb") is not None else "-"
        loc = m.get("local_path") or "-"
        lines.append(f"| `{m['model_key']}` | `{m['provider']}` | `{m['status']}` | {sz} | `{loc}` |")

    report_path = OUTPUT_DIR / "relatorio_orquestrador_prefect.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


@task(name="06_empacotar_publicacao", retries=1, retry_delay_seconds=5)
def task_06_empacotar_publicacao(run_id: str) -> Path:
    """Valida vídeos finais via ffprobe, cria output/prontos/ e atualiza tabela publicacoes."""
    logger = get_run_logger()
    db.log_pipeline_step(run_id, FLOW_NAME, "06_empacotar_publicacao", "running")
    try:
        out = _run_module(
            [sys.executable, "publicador.py", "--shortcodes", *TARGET_SHORTCODES],
            "06_empacotar_publicacao",
        )
        logger.info(out)
        pubs = db.get_publicacoes(TARGET_SHORTCODES)
        db.log_pipeline_step(
            run_id,
            FLOW_NAME,
            "06_empacotar_publicacao",
            "completed",
            detalhes=f"{len(pubs)} pacotes de publicacao validados em output/prontos/.",
        )
        report_path = export_orchestrator_report(run_id)
        logger.info(f"Relatorio consolidado gerado: {report_path}")
        return report_path
    except Exception as exc:
        db.log_pipeline_step(run_id, FLOW_NAME, "06_empacotar_publicacao", "failed", detalhes=str(exc))
        raise


@flow(name=FLOW_NAME)
def linha_de_montagem_viral(
    stage: str = "all",
    force_render: bool = False,
) -> dict[str, Any]:
    """Pipeline completo ou parcial da fábrica local de vídeos virais."""
    db.init_db()
    run_id = _current_run_id()
    logger = get_run_logger()
    logger.info(f"Iniciando {FLOW_NAME} (run_id={run_id}, stage={stage})")

    summary: dict[str, Any] = {"run_id": run_id, "stage": stage}

    if stage in ("all", "modelos"):
        summary["modelos"] = len(task_01_verificar_modelos(run_id))

    if stage in ("all", "minerar"):
        summary["mineracao"] = task_02_minerar_referencias(run_id)

    if stage in ("all", "llama"):
        summary["llama"] = task_03_diretor_llama(run_id)

    if stage in ("all", "render"):
        summary["renderizados"] = task_04_renderizar_videos_gpu(
            run_id, force_render=force_render
        )

    if stage in ("all", "masterizar"):
        summary["previews_asmr"] = task_05_masterizar_asmr(run_id)

    if stage in ("all", "publicar"):
        report = task_06_empacotar_publicacao(run_id)
        summary["relatorio"] = str(report)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Orquestrador Prefect da linha de montagem local de vídeos virais."
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Executa todas as 6 etapas da linha de montagem ponta a ponta (padrao).",
    )
    parser.add_argument(
        "--stage",
        choices=["all", "modelos", "minerar", "llama", "render", "masterizar", "publicar"],
        default="all",
        help="Executa apenas uma etapa especifica ou todas (padrao: all).",
    )
    parser.add_argument(
        "--force-render",
        action="store_true",
        help="Forca a re-renderizacao dos 4 videos em criados/ na RX 9070 XT.",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Inicia o agendador continuo do Prefect com cron definido em --cron.",
    )
    parser.add_argument(
        "--cron",
        type=str,
        default="0 12,19 * * *",
        help="Expressao cron usada com --serve (padrao: '0 12,19 * * *').",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.serve:
        print(f"Iniciando Prefect serve para '{FLOW_NAME}' com cron '{args.cron}'...")
        linha_de_montagem_viral.serve(
            name="fabrica-videos-virais-diaria",
            cron=args.cron,
        )
        return 0

    stage = "all" if args.run_all else args.stage
    result = linha_de_montagem_viral(stage=stage, force_render=args.force_render)
    print("\n=== Execucao Prefect Concluida ===")
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
