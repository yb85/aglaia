; Inno Setup installer for Aglaïa (Windows x64).
;
; Compile (version is read from the AGLAIA_VERSION env var, set by release CI
; from the git tag; falls back to 0.0.0-dev for local builds):
;   iscc packaging\aglaia.iss
; or, full path:
;   "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" packaging\aglaia.iss
;
; Input : dist\Aglaia\            (PyInstaller `Aglaia.spec` COLLECT onedir)
; Output: dist\Aglaia-windows-x64-setup.exe
;
; What it does beyond copying files: Start-menu (+ optional desktop) shortcut,
; and registers the `.agl` document type so double-clicking a project opens it
; in Aglaïa — the Windows analogue of the macOS Info.plist CFBundleDocumentTypes.

#define AppName "Aglaïa"
#define AppExe "Aglaia.exe"
#define AppPublisher "bibli.cc"
#define AppURL "https://aglaia.bibli.cc"
#define AppVer GetEnv("AGLAIA_VERSION")
#if AppVer == ""
  #define AppVer "0.0.0-dev"
#endif

[Setup]
; A stable, Aglaïa-specific GUID — never reuse for another product.
AppId={{A1E5F3C2-6B4D-4E8A-9C7F-2D1B0A9E7C44}
AppName={#AppName}
AppVersion={#AppVer}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
DefaultDirName={autopf}\Aglaia
DefaultGroupName={#AppName}
UninstallDisplayIcon={app}\{#AppExe}
OutputDir=..\dist
OutputBaseFilename=Aglaia-windows-x64-setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=admin

[Languages]
Name: "en"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; The whole PyInstaller onedir (Aglaia.exe + _internal\).
Source: "..\dist\Aglaia\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
; Document icon used by the .agl association below.
Source: "..\aglaia\assets\app\AglaiaDoc.ico"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{group}\{cm:UninstallProgram,{#AppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Registry]
; `.agl` → Aglaia.Project, opened by the app, with the document icon. HKA maps
; to HKLM for an admin (all-users) install, HKCU otherwise — so the uninstaller
; cleans up whichever hive it wrote.
Root: HKA; Subkey: "Software\Classes\.agl"; ValueType: string; ValueName: ""; ValueData: "Aglaia.Project"; Flags: uninsdeletevalue
Root: HKA; Subkey: "Software\Classes\Aglaia.Project"; ValueType: string; ValueName: ""; ValueData: "Aglaïa project"; Flags: uninsdeletekey
Root: HKA; Subkey: "Software\Classes\Aglaia.Project\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\AglaiaDoc.ico"
Root: HKA; Subkey: "Software\Classes\Aglaia.Project\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#AppExe}"" ""%1"""

[Run]
Filename: "{app}\{#AppExe}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent
