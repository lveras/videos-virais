param(
    [string]$TargetDir = "C:\videos-virais",
    [string]$RepoUrl = "https://github.com/lveras/videos-virais.git",
    [switch]$UpdateRepo,
    [switch]$CheckOnly,
    [switch]$DownloadWan21,
    [switch]$RunPipeline
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path (Join-Path $TargetDir "install_setup.ps1"))) {
    if ($CheckOnly) {
        throw "Projeto nao encontrado em '$TargetDir'. Execute sem -CheckOnly para clonar de $RepoUrl."
    }
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        throw "Git nao encontrado no PATH. Instale com: winget install Git.Git"
    }
    Write-Host "Clonando $RepoUrl -> $TargetDir..." -ForegroundColor Cyan
    git clone $RepoUrl $TargetDir
} elseif ($UpdateRepo -and (Test-Path (Join-Path $TargetDir ".git"))) {
    Write-Host "Atualizando repositorio em $TargetDir..." -ForegroundColor Cyan
    git -C $TargetDir pull --ff-only
}

$setupScript = Join-Path $TargetDir "install_setup.ps1"
$passArgs = @{
    TargetDir = $TargetDir
    RepoUrl   = $RepoUrl
}
if ($CheckOnly)     { $passArgs.CheckOnly = $true }
if ($DownloadWan21) { $passArgs.DownloadWan21 = $true }
if ($RunPipeline)   { $passArgs.RunPipeline = $true }

& $setupScript @passArgs
