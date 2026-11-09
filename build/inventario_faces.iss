#define MyAppName "Inventario Faces"
#ifndef MyAppVersion
  #define MyAppVersion "0.2.7"
#endif
#define MyAppPublisher "Laboratorio de Analise Digital"
#define MyAppURL "https://github.com/demusis/inventario_faces"
#define MyAppExeName "InventarioFaces.exe"
; Pasta gerada pelo PyInstaller; o build.ps1 informa uma pasta local via /DAppSourceDir.
#ifndef AppSourceDir
  #define AppSourceDir "..\dist\InventarioFaces"
#endif

[Setup]
AppId={{E75E8D20-2E13-4F9B-B36E-40E19493A1A8}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
; Instalacao sem privilegios de administrador: por padrao instala apenas para o usuario
; atual ({autopf} -> %LOCALAPPDATA%\Programs). Um administrador ainda pode optar, no
; dialogo inicial, por instalar para todos os usuarios.
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
UsedUserAreasWarning=no
DefaultDirName={autopf}\InventarioFaces
DefaultGroupName={#MyAppName}
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
SetupIconFile=..\src\inventario_faces\assets\app_icon.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
OutputDir=..\dist\installer
OutputBaseFilename=InventarioFacesSetup-{#MyAppVersion}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern

[Languages]
Name: "brazilianportuguese"; MessagesFile: "compiler:Languages\BrazilianPortuguese.isl"

[Files]
Source: "{#AppSourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Abrir {#MyAppName}"; Flags: nowait postinstall skipifsilent
