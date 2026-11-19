param(
    [switch]$CheckOnly,
    [string]$TargetDir = "C:\videos-virais",
    [string]$RepoUrl = "https://github.com/lveras/videos-virais.git",
    [switch]$DownloadWan21,
    [switch]$RunPipeline
)

$ErrorActionPreference = "Stop"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  Fabrica Local de Videos Virais ASMR - Setup & Check" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

function Test-Tool {
    param(
        [string]$Cmd,
        [string]$WingetHint
    )
    $found = Get-Command $Cmd -ErrorAction SilentlyContinue
    if ($found) {
        Write-Host "[OK]   $Cmd -> $($found.Source)" -ForegroundColor Green
        return $true
    } else {
        Write-Host "[FALTA] $Cmd nao encontrado no PATH. Instale com: $WingetHint" -ForegroundColor Yellow
        return $false
    }
}

$hasGit    = Test-Tool -Cmd "git"    -WingetHint "winget install Git.Git"
$hasPython = Test-Tool -Cmd "python" -WingetHint "winget install Python.Python.3.11"
$hasFfmpeg = Test-Tool -Cmd "ffmpeg" -WingetHint "winget install Gyan.FFmpeg"
$hasYtdlp  = Test-Tool -Cmd "yt-dlp" -WingetHint "python -m pip install yt-dlp"
$hasOllama = Test-Tool -Cmd "ollama" -WingetHint "winget install Ollama.Ollama"

if (-not $hasPython) {
    throw "Python e obrigatorio para continuar."
}

# Resolve pasta do projeto (clona se nao existir localmente e nao for CheckOnly)
$ScriptRoot = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }
$ProjectDir = if (Test-Path (Join-Path $ScriptRoot "orquestrador_prefect.py")) { $ScriptRoot } else { $TargetDir }

if (-not (Test-Path (Join-Path $ProjectDir "orquestrador_prefect.py"))) {
    if ($CheckOnly) {
        throw "Projeto nao encontrado em '$ProjectDir'. Rode sem -CheckOnly para clonar de $RepoUrl."
    }
    if (-not $hasGit) {
        throw "Git e obrigatorio para clonar $RepoUrl."
    }
    Write-Host "`nClonando repositorio para $ProjectDir..." -ForegroundColor Cyan
    git clone $RepoUrl $ProjectDir
}

Push-Location $ProjectDir
try {
    $dirs = @(
        "baixados",
        "criados",
        "data",
        "modelos\wan21",
        "modelos\text_encoders",
        "modelos\vae",
        "output\prontos",
        "output\publicados",
        "referencias\videos",
        "referencias\audios_asmr",
        "referencias\frames",
        "workflows",
        "config"
    )
    foreach ($d in $dirs) {
        $full = Join-Path $ProjectDir $d
        if (-not (Test-Path $full)) {
            if (-not $CheckOnly) {
                New-Item -ItemType Directory -Path $full -Force | Out-Null
            }
        }
    }
    Write-Host "[OK]   Estrutura de pastas validada em $ProjectDir" -ForegroundColor Green

    if (-not $CheckOnly) {
        Write-Host "`nInstalando dependencias Python (requirements.txt)..." -ForegroundColor Cyan
        python -m pip install -r (Join-Path $ProjectDir "requirements.txt")

        if ($hasOllama) {
            $llamaManifest = Join-Path $env:USERPROFILE ".ollama\models\manifests\registry.ollama.ai\library\llama3.1\latest"
            if (-not (Test-Path $llamaManifest)) {
                Write-Host "`nBaixando modelo llama3.1:latest via Ollama..." -ForegroundColor Cyan
                ollama pull llama3.1:latest
            } else {
                Write-Host "[OK]   Modelo llama3.1:latest ja presente no cache Ollama." -ForegroundColor Green
            }
        }

        # Instala/atualiza skill globalmente no Antigravity (~/.gemini/config/skills/)
        $repoSkillDir = Join-Path $ProjectDir ".agents\skills\videos-virais-installer"
        $globalSkillDir = Join-Path $env:USERPROFILE ".gemini\config\skills\videos-virais-installer"
        if (Test-Path $repoSkillDir) {
            New-Item -ItemType Directory -Path $globalSkillDir -Force | Out-Null
            Copy-Item -Path (Join-Path $repoSkillDir "*") -Destination $globalSkillDir -Recurse -Force
            Write-Host "[OK]   Skill instalada globalmente em $globalSkillDir" -ForegroundColor Green
        }
    } else {
        Write-Host "`n[CheckOnly] Verificando modulos Python essenciais..." -ForegroundColor Cyan
        python -c "import prefect, huggingface_hub, requests, playwright; print('[OK]   Prefect', prefect.__version__, '+ pacotes do orquestrador OK')"
        if ($LASTEXITCODE -ne 0) { throw "Modulos Python do orquestrador ausentes. Rode install_setup.ps1 sem -CheckOnly." }
        $rocmPy = Join-Path $env:LOCALAPPDATA "AMD\AI_Bundle\ComfyUI\venv\Scripts\python.exe"
        if (Test-Path $rocmPy) {
            & $rocmPy -c "import torch, numpy, scipy; print('[OK]   ROCm PyTorch', torch.__version__, '| GPU:', torch.cuda.is_available())"
        }
    }

    Write-Host "`nExecutando inventario de modelos (gerenciador_modelos.py --action check)..." -ForegroundColor Cyan
    python (Join-Path $ProjectDir "gerenciador_modelos.py") --action check

    if (-not $CheckOnly -and $DownloadWan21) {
        foreach ($key in @("wan21_1_3b_t2v", "wan21_vae", "umt5_xxl_fp8")) {
            Write-Host "`nBaixando peso Wan 2.1: $key..." -ForegroundColor Cyan
            python (Join-Path $ProjectDir "gerenciador_modelos.py") --action download --model-key $key
        }
    }

    if (-not $CheckOnly -and $RunPipeline) {
        Write-Host "`nExecutando linha de montagem completa no Prefect..." -ForegroundColor Cyan
        python (Join-Path $ProjectDir "orquestrador_prefect.py") --run-all
    }

    Write-Host "`nSetup/Verificacao finalizado com sucesso." -ForegroundColor Green
} finally {
    Pop-Location
}
