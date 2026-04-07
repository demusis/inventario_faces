param(
    [string]$PythonExe = "python",
    [string]$BuildPythonExe = "",
    [switch]$SkipTests,
    [switch]$SkipInstaller
)

$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ProjectVenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$VenvPath = Join-Path $ProjectRoot ".venv-build"
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"
$SpecPath = Join-Path $PSScriptRoot "inventario_faces.spec"
$InstallerScript = Join-Path $PSScriptRoot "inventario_faces.iss"
$SourceRoot = (Join-Path $ProjectRoot "src").Replace("\", "/")
$VersionScript = "import sys; sys.path.insert(0, r'$SourceRoot'); from inventario_faces import __version__; print(__version__)"

Write-Host "Projeto:" $ProjectRoot

if ($BuildPythonExe) {
    $SelectedPython = $BuildPythonExe
    Write-Host "Usando interpretador informado para build:" $SelectedPython
}
elseif (Test-Path $ProjectVenvPython) {
    $SelectedPython = $ProjectVenvPython
    Write-Host "Reutilizando ambiente virtual do projeto:" $SelectedPython
}
else {
    Write-Host "Criando ambiente virtual de build em" $VenvPath
    & $PythonExe -m venv $VenvPath
    $SelectedPython = $VenvPython
}

& $SelectedPython -m pip install --upgrade pip
& $SelectedPython -m pip install -r (Join-Path $ProjectRoot "requirements-dev.txt")
$VenvPython = $SelectedPython
$AppVersion = (& $VenvPython -c $VersionScript).Trim()
Write-Host "Versao detectada:" $AppVersion

if (-not $SkipTests) {
    Write-Host "Executando testes..."
    Push-Location $ProjectRoot
    try {
        & $VenvPython -m unittest discover -s tests -v
    }
    finally {
        Pop-Location
    }
}

Write-Host "Gerando executavel com PyInstaller..."
Push-Location $ProjectRoot
try {
    & $VenvPython -m PyInstaller $SpecPath --noconfirm
}
finally {
    Pop-Location
}

if ($SkipInstaller) {
    Write-Host "Build do instalador ignorado."
    exit 0
}

$InnoCandidates = @(
    @(
        (Get-Command "iscc.exe" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -ErrorAction SilentlyContinue),
        "C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
        "C:\Program Files\Inno Setup 6\ISCC.exe",
        (Join-Path $env:LOCALAPPDATA "Programs\Inno Setup 6\ISCC.exe")
    ) | Where-Object { $_ -and (Test-Path $_) }
)

if ($InnoCandidates.Count -eq 0) {
    Write-Warning "ISCC.exe nao encontrado. O executavel foi gerado, mas o instalador nao foi compilado."
    exit 0
}

$IsccPath = $InnoCandidates[0]
Write-Host "Compilando instalador com" $IsccPath
& $IsccPath "/DMyAppVersion=$AppVersion" $InstallerScript
