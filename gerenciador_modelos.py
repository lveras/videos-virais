"""Módulo independente 1: gestor de modelos/runtimes locais, pastas organizadas e catálogo SQLite."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import db

BASE_DIR = Path(__file__).resolve().parent
MODELOS_DIR = BASE_DIR / "modelos"
WAN21_DIR = MODELOS_DIR / "wan21"
TEXT_ENC_DIR = MODELOS_DIR / "text_encoders"
VAE_DIR = MODELOS_DIR / "vae"

COMFY_BASE = Path(r"C:\Users\lucia\AppData\Local\AMD\AI_Bundle\ComfyUI")
COMFY_APP_DIR = COMFY_BASE / "ComfyUI"
COMFY_PYTHON = COMFY_BASE / "venv" / "Scripts" / "python.exe"
OLLAMA_MANIFEST_PATH = (
    Path.home()
    / ".ollama"
    / "models"
    / "manifests"
    / "registry.ollama.ai"
    / "library"
    / "llama3.1"
    / "latest"
)
OLLAMA_BLOBS_DIR = Path.home() / ".ollama" / "models" / "blobs"

MODEL_SPECS: list[dict[str, Any]] = [
    {
        "model_key": "llama3.1:latest",
        "nome": "Llama 3.1 8B (Ollama Diretor Criativo)",
        "provider": "ollama",
        "repo_id": "registry.ollama.ai/library/llama3.1:latest",
        "filename": "latest",
        "target_dir": str(OLLAMA_MANIFEST_PATH.parent),
        "obrigatorio": 1,
    },
    {
        "model_key": "pytorch_rocm_rx9070xt",
        "nome": "PyTorch ROCm SDK (AMD Radeon RX 9070 XT)",
        "provider": "pytorch_rocm",
        "repo_id": "AMD AI Bundle ROCm venv",
        "filename": "python.exe",
        "target_dir": str(COMFY_PYTHON.parent),
        "obrigatorio": 1,
    },
    {
        "model_key": "comfyui_amd_bundle",
        "nome": "ComfyUI AMD Bundle (Engine Video/Diffusion)",
        "provider": "comfyui",
        "repo_id": "AMD AI Bundle ComfyUI",
        "filename": "main.py",
        "target_dir": str(COMFY_APP_DIR),
        "obrigatorio": 1,
    },
    {
        "model_key": "wan21_1_3b_t2v",
        "nome": "Wan 2.1 T2V 1.3B (Difusão de Vídeo)",
        "provider": "huggingface",
        "repo_id": "Wan-AI/Wan2.1-T2V-1.3B",
        "filename": "diffusion_pytorch_model.safetensors",
        "target_dir": "modelos/wan21",
        "comfy_subdirs": ["models/diffusion_models", "models/unet"],
        "obrigatorio": 0,
    },
    {
        "model_key": "wan21_vae",
        "nome": "Wan 2.1 VAE",
        "provider": "huggingface",
        "repo_id": "Wan-AI/Wan2.1-T2V-1.3B",
        "filename": "Wan2.1_VAE.pth",
        "target_dir": "modelos/vae",
        "comfy_subdirs": ["models/vae"],
        "obrigatorio": 0,
    },
    {
        "model_key": "umt5_xxl_fp8",
        "nome": "UMT5 XXL FP8 Scaled Text Encoder",
        "provider": "huggingface",
        "repo_id": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
        "filename": "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        "target_dir": "modelos/text_encoders",
        "comfy_subdirs": ["models/text_encoders", "models/clip"],
        "obrigatorio": 0,
    },
]


def ensure_model_dirs() -> None:
    """Garante a estrutura de diretórios de modelos locais."""
    for folder in (MODELOS_DIR, WAN21_DIR, TEXT_ENC_DIR, VAE_DIR):
        folder.mkdir(parents=True, exist_ok=True)


def write_extra_model_paths_yaml() -> Path:
    """Gera modelos/extra_model_paths.yaml apontando os diretórios locais ao ComfyUI."""
    ensure_model_dirs()
    yaml_path = MODELOS_DIR / "extra_model_paths.yaml"
    base_posix = MODELOS_DIR.resolve().as_posix()
    content = (
        "# Gerado automaticamente por gerenciador_modelos.py\n"
        "# Para ativar no ComfyUI, passe --extra-model-paths-config ou copie para a raiz do ComfyUI.\n"
        "videos_virais_local:\n"
        f"  base_path: {base_posix}\n"
        "  diffusion_models: wan21/\n"
        "  unet: wan21/\n"
        "  text_encoders: |\n"
        "    text_encoders/\n"
        "    text_encoders/split_files/text_encoders/\n"
        "  clip: |\n"
        "    text_encoders/\n"
        "    text_encoders/split_files/text_encoders/\n"
        "  vae: vae/\n"
    )
    yaml_path.write_text(content, encoding="utf-8")
    return yaml_path


def _ollama_api_has_model(model_name: str) -> bool:
    try:
        req = urllib.request.Request("http://127.0.0.1:11434/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            names = [m.get("name", "") for m in payload.get("models", [])]
            return model_name in names
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return False


def _estimate_ollama_manifest_size_mb(manifest_path: Path) -> float | None:
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        total_bytes = 0
        for layer in data.get("layers", []):
            total_bytes += int(layer.get("size") or 0)
        cfg = data.get("config") or {}
        total_bytes += int(cfg.get("size") or 0)
        if total_bytes > 0:
            return round(total_bytes / (1024 * 1024), 2)
    except Exception:
        pass
    return None


def inspect_model(spec: dict[str, Any]) -> dict[str, Any]:
    """Verifica a disponibilidade local de um modelo ou runtime e retorna o registro."""
    provider = spec["provider"]
    key = spec["model_key"]
    status = "ausente"
    local_path: str | None = None
    size_mb: float | None = None

    if provider == "ollama":
        if OLLAMA_MANIFEST_PATH.exists():
            status = "instalado"
            local_path = str(OLLAMA_MANIFEST_PATH)
            size_mb = _estimate_ollama_manifest_size_mb(OLLAMA_MANIFEST_PATH)
        elif _ollama_api_has_model(key):
            status = "instalado"
            local_path = "ollama://127.0.0.1:11434"

    elif provider == "pytorch_rocm":
        if COMFY_PYTHON.exists():
            status = "instalado"
            local_path = str(COMFY_PYTHON)
            size_mb = round(COMFY_PYTHON.stat().st_size / (1024 * 1024), 2)

    elif provider == "comfyui":
        main_py = COMFY_APP_DIR / "main.py"
        if main_py.exists():
            status = "instalado"
            local_path = str(main_py)
            size_mb = round(main_py.stat().st_size / (1024 * 1024), 2)

    elif provider == "huggingface":
        rel_dir = BASE_DIR / spec["target_dir"]
        filename = spec["filename"]
        basename = Path(filename).name
        candidates = [
            rel_dir / filename,
            rel_dir / basename,
        ]
        for sub in spec.get("comfy_subdirs", []):
            candidates.append(COMFY_APP_DIR / sub / basename)
            candidates.append(COMFY_APP_DIR / sub / filename)

        found = next((c for c in candidates if c.exists() and c.stat().st_size > 0), None)
        if found:
            status = "instalado"
            local_path = str(found)
            size_mb = round(found.stat().st_size / (1024 * 1024), 2)

    return {
        "model_key": key,
        "nome": spec["nome"],
        "provider": provider,
        "repo_id": spec.get("repo_id"),
        "filename": spec.get("filename"),
        "target_dir": spec.get("target_dir"),
        "local_path": local_path,
        "size_mb": size_mb,
        "obrigatorio": spec.get("obrigatorio", 1),
        "status": status,
    }


def check_all_models() -> list[dict[str, Any]]:
    """Verifica todos os modelos/runtimes, salva no SQLite e exporta catalogo_modelos.json."""
    ensure_model_dirs()
    db.init_db()
    write_extra_model_paths_yaml()

    results: list[dict[str, Any]] = []
    for spec in MODEL_SPECS:
        record = inspect_model(spec)
        db.upsert_modelo_local(record)
        results.append(record)

    catalog_path = MODELOS_DIR / "catalogo_modelos.json"
    catalog_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return results


def download_model(model_key: str) -> dict[str, Any]:
    """Baixa sob demanda o modelo especificado para a pasta local organizada."""
    ensure_model_dirs()
    db.init_db()

    spec = next((s for s in MODEL_SPECS if s["model_key"] == model_key), None)
    if not spec:
        valid = ", ".join(s["model_key"] for s in MODEL_SPECS)
        raise ValueError(f"model_key desconhecido '{model_key}'. Opcoes validas: {valid}")

    provider = spec["provider"]
    if provider == "ollama":
        print(f"Executando: ollama pull {model_key} ...")
        res = subprocess.run(["ollama", "pull", model_key], check=False)
        if res.returncode != 0:
            raise RuntimeError(f"`ollama pull {model_key}` falhou com codigo {res.returncode}")

    elif provider == "huggingface":
        from huggingface_hub import hf_hub_download

        target_dir = BASE_DIR / spec["target_dir"]
        target_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"Baixando {spec['repo_id']} / {spec['filename']} -> {target_dir.relative_to(BASE_DIR).as_posix()}..."
        )
        downloaded_path = hf_hub_download(
            repo_id=spec["repo_id"],
            filename=spec["filename"],
            local_dir=str(target_dir),
        )
        print(f"Download concluido: {downloaded_path}")

    else:
        print(f"[INFO] Runtime '{model_key}' ({provider}) e gerenciado pelo AMD AI Bundle.")

    record = inspect_model(spec)
    db.upsert_modelo_local(record)
    check_all_models()
    return record


def print_report(records: list[dict[str, Any]]) -> None:
    """Imprime relatório legível dos modelos locais."""
    print("=== Inventario de Modelos e Runtimes Locais ===")
    for r in records:
        obrig = "OBRIGATORIO" if r.get("obrigatorio") else "OPCIONAL   "
        size_txt = f"{r['size_mb']:.1f} MB" if r.get("size_mb") is not None else "-"
        loc = r.get("local_path") or "(nao baixado)"
        print(f"[{r['status'].upper():9}] [{obrig}] {r['model_key']:24} | {size_txt:>10} | {loc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gerenciador independente de modelos e runtimes locais (SQLite + pastas modelos/)."
    )
    parser.add_argument(
        "--action",
        choices=["check", "download"],
        default="check",
        help="Acao a executar: check (padrao) ou download.",
    )
    parser.add_argument(
        "--model-key",
        type=str,
        default=None,
        help="Chave do modelo para baixar quando --action download for usado.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.action == "download":
            if not args.model_key:
                print("[ERRO] --model-key e obrigatorio com --action download.", file=sys.stderr)
                return 1
            rec = download_model(args.model_key)
            print(f"\nStatus atualizado para {rec['model_key']}: {rec['status']} ({rec.get('local_path')})")
        else:
            records = check_all_models()
            print_report(records)
            print(f"\nCatalogo salvo em: {(MODELOS_DIR / 'catalogo_modelos.json').relative_to(BASE_DIR).as_posix()}")
            print(f"ComfyUI paths em:  {(MODELOS_DIR / 'extra_model_paths.yaml').relative_to(BASE_DIR).as_posix()}")
    except Exception as exc:
        print(f"[ERRO] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
