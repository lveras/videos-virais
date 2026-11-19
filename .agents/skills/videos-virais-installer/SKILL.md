---
name: videos-virais-installer
description: >-
  Instala, restaura do GitHub (lveras/videos-virais), valida dependências e configura
  a linha de montagem local de vídeos virais ASMR (Prefect, Llama 3.1 via Ollama Vulkan,
  PyTorch ROCm RX 9070 XT, Wan 2.1 e SQLite). Use esta skill sempre que o usuário pedir
  para instalar, clonar, restaurar, configurar em outra máquina ou reparar o projeto videos-virais.
---

# Instalador da Fábrica Local de Vídeos Virais ASMR (`lveras/videos-virais`)

Esta skill guia o agente na instalação, restauração do GitHub, validação de dependências, download sob demanda de modelos de IA e execução do pipeline Prefect do projeto **videos-virais**.

## Locais do Projeto e Repositório

- **Repositório GitHub:** `https://github.com/lveras/videos-virais.git`
- **Diretório padrão no Windows:** `C:\videos-virais`
- **Banco de Conhecimento SQLite:** `C:\videos-virais\data\base_conhecimento.db`
- **Script de Setup Principal:** `C:\videos-virais\install_setup.ps1`
- **Script Auxiliar desta Skill:** `scripts\install_and_verify.ps1` (dentro desta pasta de skill)

---

## Procedimento de Instalação Passo a Passo

### 1. Clonar ou atualizar o repositório em `C:\videos-virais`
Se a pasta `C:\videos-virais` ainda não existir, clone do GitHub:
```powershell
git clone https://github.com/lveras/videos-virais.git C:\videos-virais
```
Se já existir, atualize quando necessário:
```powershell
git -C C:\videos-virais pull --ff-only
```

### 2. Executar o instalador automático (`install_setup.ps1` ou `scripts/install_and_verify.ps1`)
O script verifica `git`, `python`, `ffmpeg`, `yt-dlp` e `ollama`, cria todas as pastas (`baixados`, `criados`, `data`, `modelos/*`, `output/prontos`, `output/publicados`, `referencias/*`, `config`), instala `requirements.txt` (`prefect`, `yt-dlp`, `huggingface_hub`, `requests`, `numpy`, `scipy`, `playwright`), garante o modelo `llama3.1:latest` no Ollama e valida `gerenciador_modelos.py --action check`.

* **Apenas diagnosticar ambiente (sem alterar pacotes):**
  ```powershell
  powershell -NoProfile -ExecutionPolicy Bypass -File "C:\videos-virais\install_setup.ps1" -CheckOnly
  ```
* **Instalação / Reparo completo:**
  ```powershell
  powershell -NoProfile -ExecutionPolicy Bypass -File "C:\videos-virais\install_setup.ps1"
  ```
* **Instalação a partir desta skill mesmo antes do clone existir:**
  ```powershell
  powershell -NoProfile -ExecutionPolicy Bypass -File "<caminho-desta-skill>\scripts\install_and_verify.ps1"
  ```

### 3. Re-hidratar vídeos de referência e áudios ASMR brutos
Como vídeos pesados baixados e `.wav` brutos ficam fora do Git (`.gitignore`), rode o pipeline Prefect para baixar novamente as referências, extrair os áudios ASMR, atualizar os roteiros no SQLite, validar os renders e gerar os pacotes em `output/prontos/`:
```powershell
cd C:\videos-virais
python orquestrador_prefect.py --run-all
```

### 4. Baixar pesos opcionais do Wan 2.1 (quando usar ComfyUI Vid2Vid)
Os pesos de difusão ficam em `C:\videos-virais\modelos\` (integrados ao ComfyUI via `modelos/extra_model_paths.yaml`):
```powershell
python gerenciador_modelos.py --action download --model-key wan21_1_3b_t2v
python gerenciador_modelos.py --action download --model-key wan21_vae
python gerenciador_modelos.py --action download --model-key umt5_xxl_fp8
```

### 5. Ativar agendamento contínuo no Prefect (opcional)
Para executar a linha de montagem automaticamente todo dia às 12h e 19h:
```powershell
python orquestrador_prefect.py --serve --cron "0 12,19 * * *"
```

---

## Pré-requisitos Externos (se for uma máquina Windows zerada)
Caso `install_setup.ps1` reporte falta de binários externos no `PATH`:
- **Git:** `winget install Git.Git`
- **Python 3.11+:** `winget install Python.Python.3.11`
- **FFmpeg:** `winget install Gyan.FFmpeg`
- **Ollama:** `winget install Ollama.Ollama`
- **AMD AI Bundle (para PyTorch ROCm + ComfyUI em GPUs Radeon):** instalar via AMD Software / AI Bundle para habilitar `C:\Users\<usuario>\AppData\Local\AMD\AI_Bundle\ComfyUI\venv\Scripts\python.exe`.
