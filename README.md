# Fábrica Local de Vídeos Virais ASMR (Prefect + Llama 3.1 + GPU ROCm + SQLite)

Linha de montagem **100% local** no Windows para minerar referências virais (estilo `@HydraulicPressChannel` e `@MagnusRugCleaning`), gerar direção criativa e prompts técnicos com **Llama 3.1** (acelerado na GPU via **Ollama Vulkan**), renderizar vídeos originais **1080×1920 @ 60fps** em GPU **AMD Radeon RX 9070 XT** (PyTorch ROCm) com áudio ASMR sincronizado e empacotar tudo pronto para **YouTube Shorts**, **Instagram Reels** e **TikTok**.

---

## 🧩 Arquitetura dos Módulos Independentes

| Etapa Prefect | Script | Responsabilidade |
| :--- | :--- | :--- |
| `01_verificar_modelos` | `gerenciador_modelos.py` | Audita runtimes/modelos (`llama3.1:latest`, PyTorch ROCm, ComfyUI, pesos Wan 2.1), gera `modelos/extra_model_paths.yaml` e grava em `modelos_locais`. |
| `02_minerar_referencias` | `minerador.py` | Baixa Reels/Shorts (`yt-dlp` com fallback `android`), extrai áudio bruto + masterizado ASMR (`FFmpeg`), captura 3 frames-chave e sincroniza `baixados/`. |
| `03_diretor_criativo_llama` | `diretor_llama.py` | Inicia `ollama serve` com `OLLAMA_VULKAN=1`, gera prompts Wan 2.1, `denoise`, títulos PT/EN e hashtags; ao final descarrega o modelo (`keep_alive=0`) liberando os 16 GB de VRAM. |
| `04_renderizar_gpu_original` | `gerador_original_gpu.py` | Renderiza 4 cenas 3D PBR/Raymarching 100% originais na RX 9070 XT com áudio ASMR físico sintetizado em estéreo (`criados/`). |
| `05_masterizar_asmr` | `masterizador.py` | Faz upscale/pad `1080x1920`, aplica realce macro e junta áudio ASMR masterizado (`--mux` para Wan 2.1 ou `--preview-remix`). |
| `06_empacotar_publicacao` | `publicador.py` | Valida codec/resolução via `ffprobe`, grava payloads multi-rede na tabela `publicacoes` e organiza `output/prontos/<shortcode>/` (`FOLHA_POSTAGEM.md` + `pacote_redes.json`). |

---

## 🗄️ Base de Conhecimento Local (`data/base_conhecimento.db`)

Todas as etapas compartilham o banco SQLite local com 5 tabelas:
- `videos_referencia`: URLs, métricas, FPS, resolução, caminhos de vídeo/áudio ASMR e frames-chave.
- `roteiros_ia`: prompts positivo/negativo Wan 2.1, `recommended_denoise`, títulos virais PT/EN e hashtags geradas pelo Llama 3.1.
- `modelos_locais`: inventário de modelos Ollama, PyTorch ROCm, ComfyUI e pesos HuggingFace.
- `pipeline_execucoes`: histórico de execuções por `run_id` do Prefect.
- `publicacoes`: pacotes validados para YouTube Shorts, Instagram Reels e TikTok.

---

## ⚡ Instalação Rápida em Nova Máquina

### Opção 1 — Script automático PowerShell
```powershell
git clone https://github.com/lveras/videos-virais.git C:\videos-virais
cd C:\videos-virais
powershell -NoProfile -ExecutionPolicy Bypass -File .\install_setup.ps1
```

Para apenas verificar dependências sem modificar o ambiente:
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\install_setup.ps1 -CheckOnly
```

### Opção 2 — Via Skill do Antigravity (`videos-virais-installer`)
O repositório inclui a skill pronta em `.agents/skills/videos-virais-installer/`. O `install_setup.ps1` também a instala globalmente em `~/.gemini/config/skills/videos-virais-installer/`.

---

## 🚀 Comandos do Dia a Dia

```powershell
# Rodar todas as 6 etapas na linha de montagem Prefect
python orquestrador_prefect.py --run-all

# Rodar apenas uma etapa isolada
python orquestrador_prefect.py --stage modelos     # ou: minerar | llama | render | masterizar | publicar

# Baixar pesos opcionais do Wan 2.1 sob demanda (para ComfyUI)
python gerenciador_modelos.py --action download --model-key wan21_1_3b_t2v
python gerenciador_modelos.py --action download --model-key wan21_vae
python gerenciador_modelos.py --action download --model-key umt5_xxl_fp8

# Minerar os Top 5 Shorts mais vistos de um canal
python minerador.py --channel-shorts https://www.youtube.com/@MagnusRugCleaning/shorts --top 5 --categoria rug_cleaning_asmr

# Subir agendador diário local do Prefect (12h e 19h)
python orquestrador_prefect.py --serve --cron "0 12,19 * * *"
```

---

## 📤 Publicação Multi-Rede

Os pacotes prontos para postagem ficam organizados em `output/prontos/<shortcode>/` (`FOLHA_POSTAGEM.md` e `pacote_redes.json`).
Para conectar as APIs oficiais no futuro, copie `config/credenciais_publicacao.example.json` para `config/credenciais_publicacao.json` e preencha suas credenciais do **YouTube Data API v3**, **Instagram Graph API** ou **TikTok Content Posting API**.
