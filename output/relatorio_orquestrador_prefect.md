# Relatório da Linha de Montagem Local (Prefect + SQLite)

- **Flow:** `linha_de_montagem_viral`
- **Run ID:** `b1c5c36f-33c9-4302-ae8b-4adba6bcad52`
- **Modelos/Runtimes registrados:** 6 (3 instalados)
- **Vídeos de referência indexados:** 14
- **Roteiros Llama 3.1 prontos:** 14
- **Publicações empacotadas (`output/prontos/`):** 12

## Execução das Etapas neste Run

| Etapa | Status | Detalhes |
| :--- | :--- | :--- |
| `01_verificar_modelos` | `running` |  |
| `01_verificar_modelos` | `completed` | 6 modelos/runtimes verificados no SQLite. |
| `02_minerar_referencias` | `running` |  |
| `02_minerar_referencias` | `completed` | 14 videos indexados; 0 sincronizados em baixados/. |
| `03_diretor_criativo_llama` | `running` |  |
| `03_diretor_criativo_llama` | `completed` | 14/14 videos com roteiro Llama 3.1 pronto e VRAM liberada. |
| `04_renderizar_gpu_original` | `running` |  |
| `04_renderizar_gpu_original` | `completed` | 4/4 videos em criados/ prontos (aguardou=185s; renderizou=[]). |
| `05_masterizar_asmr` | `running` |  |
| `05_masterizar_asmr` | `completed` | Previews ASMR validados: FDa1Oa2y0NY_preview_asmr_remix.mp4, DfNRXxjou-k_preview_asmr_remix.mp4, Df6LIe7uxxw_preview_asmr_remix.mp4, uhQXBMZ-sTg_preview_asmr_remix.mp4 |
| `06_empacotar_publicacao` | `running` |  |
| `06_empacotar_publicacao` | `completed` | 12 pacotes de publicacao validados em output/prontos/. |

## Inventário de Modelos (`modelos_locais`)

| Chave | Provider | Status | Tamanho (MB) | Caminho |
| :--- | :--- | :--- | ---: | :--- |
| `comfyui_amd_bundle` | `comfyui` | `instalado` | 0.0 | `C:\Users\lucia\AppData\Local\AMD\AI_Bundle\ComfyUI\ComfyUI\main.py` |
| `llama3.1:latest` | `ollama` | `instalado` | 4692.8 | `C:\Users\lucia\.ollama\models\manifests\registry.ollama.ai\library\llama3.1\latest` |
| `pytorch_rocm_rx9070xt` | `pytorch_rocm` | `instalado` | 0.1 | `C:\Users\lucia\AppData\Local\AMD\AI_Bundle\ComfyUI\venv\Scripts\python.exe` |
| `umt5_xxl_fp8` | `huggingface` | `ausente` | - | `-` |
| `wan21_1_3b_t2v` | `huggingface` | `ausente` | - | `-` |
| `wan21_vae` | `huggingface` | `ausente` | - | `-` |