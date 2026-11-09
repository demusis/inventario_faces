param(
    [string]$PythonExe = "python",
    [string]$BuildPythonExe = "",
    # Pasta local (fora de Google Drive/OneDrive) para os artefatos intermediarios do
    # PyInstaller: milhares de arquivos que o cliente de sincronizacao trava e tenta enviar.
    [string]$BuildRoot = (Join-Path $env:LOCALAPPDATA "inventario_faces_build"),
    # Saida do instalador (~1,5 GB). Fica fora da pasta sincronizada por padrao: o
    # cliente do Google Drive chegou a remover o .exe recem-gerado em dist\installer.
    [string]$InstallerOutput = "",
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

# No Windows PowerShell 5.1, falhas de executaveis nativos (pip, PyInstaller, ISCC)
# nao disparam $ErrorActionPreference; sem esta verificacao o build seguiria adiante
# e empacotaria um executavel quebrado.
function Invoke-Native {
    param([string]$Description, [scriptblock]$Command)
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Description falhou (codigo de saida $LASTEXITCODE)."
    }
}

function Test-Interpreter {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        return $false
    }
    # Um venv copiado de outra maquina aponta para um Python inexistente.
    & $Path -c "import sys" 2>$null
    return ($LASTEXITCODE -eq 0)
}

Write-Host "Projeto:" $ProjectRoot

if ($BuildPythonExe) {
    $SelectedPython = $BuildPythonExe
    Write-Host "Usando interpretador informado para build:" $SelectedPython
}
elseif (Test-Interpreter $ProjectVenvPython) {
    $SelectedPython = $ProjectVenvPython
    Write-Host "Reutilizando ambiente virtual do projeto:" $SelectedPython
}
else {
    if (-not (Test-Interpreter $VenvPython)) {
        Write-Host "Criando ambiente virtual de build em" $VenvPath
        Invoke-Native "Criacao do ambiente virtual" { & $PythonExe -m venv --clear $VenvPath }
    }
    $SelectedPython = $VenvPython
}

Invoke-Native "Atualizacao do pip" { & $SelectedPython -m pip install --upgrade pip }
Invoke-Native "Instalacao das dependencias" { & $SelectedPython -m pip install -r (Join-Path $ProjectRoot "requirements-dev.txt") }
$VenvPython = $SelectedPython
$AppVersion = (& $VenvPython -c $VersionScript).Trim()
Write-Host "Versao detectada:" $AppVersion

if (-not $SkipTests) {
    Write-Host "Executando testes..."
    Push-Location $ProjectRoot
    try {
        Invoke-Native "Suite de testes" { & $VenvPython -m unittest discover -s tests -v }
    }
    finally {
        Pop-Location
    }
}

$DistPath = Join-Path $BuildRoot "dist"
$WorkPath = Join-Path $BuildRoot "work"
if (-not $InstallerOutput) {
    $InstallerOutput = Join-Path $BuildRoot "installer"
}
New-Item -ItemType Directory -Force -Path $DistPath, $WorkPath, $InstallerOutput | Out-Null

Write-Host "Gerando executavel com PyInstaller em" $DistPath
Push-Location $ProjectRoot
try {
    Invoke-Native "PyInstaller" { & $VenvPython -m PyInstaller $SpecPath --noconfirm --distpath $DistPath --workpath $WorkPath }
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
$AppSourceDir = Join-Path $DistPath "InventarioFaces"
Invoke-Native "Compilacao do instalador" {
    & $IsccPath "/DMyAppVersion=$AppVersion" "/DAppSourceDir=$AppSourceDir" "/O$InstallerOutput" $InstallerScript
}
Write-Host "Instalador gerado em" (Join-Path $InstallerOutput "InventarioFacesSetup-$AppVersion.exe")
