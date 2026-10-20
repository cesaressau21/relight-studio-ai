; Inno Setup script for Relight Studio AI (lightweight installer).
; Build with packaging\build_installer.py, or:
;   ISCC.exe /DAppVersion=0.1.0 packaging\installer\relight_studio.iss
;
; The setup.exe holds only the application source (a few MB). After copying it, setup runs
; install.ps1, which downloads uv, Python, PyTorch (CUDA build chosen for the GPU) and the
; other packages into the installation folder.

#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif
#ifndef AppVersionLabel
  #define AppVersionLabel AppVersion + "-beta"
#endif
#define Root "..\.."

[Setup]
AppId={{11E15853-E9BC-43C4-B305-2AD04DF5DFF4}
AppName=Relight Studio AI
AppVersion={#AppVersion}
AppVerName=Relight Studio AI {#AppVersionLabel}
AppPublisher=Shen
AppPublisherURL=https://github.com/cesaressau21/relight-studio-ai
AppSupportURL=https://github.com/cesaressau21/relight-studio-ai/issues
DefaultDirName={autopf}\Relight Studio AI
DefaultGroupName=Relight Studio AI
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
; RedirectionGuard (Inno 6.5+) is inherited by install.ps1 and stops uv from using the
; directory junction it creates for the Python minor version (os error 448). It guards
; elevated installers against junction attacks; this one never runs elevated.
RedirectionGuard=no
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
LicenseFile={#Root}\LICENSE
SetupIconFile={#Root}\assets\icon.ico
UninstallDisplayIcon={app}\app\assets\icon.ico
WizardStyle=modern
Compression=lzma2/max
SolidCompression=yes
OutputDir={#Root}\dist\installer
OutputBaseFilename=RelightStudioAI-{#AppVersionLabel}-setup

[Messages]
ReadyLabel2a=Setup will copy Relight Studio AI and then download its components (about 3 GB with an NVIDIA GPU, less without one). Keep the computer online until it finishes.

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "{#Root}\relight_studio\*"; DestDir: "{app}\app\relight_studio"; Excludes: "__pycache__,*.pyc"; Flags: recursesubdirs ignoreversion
Source: "{#Root}\presets\*"; DestDir: "{app}\app\presets"; Flags: recursesubdirs ignoreversion
Source: "{#Root}\assets\*"; DestDir: "{app}\app\assets"; Flags: recursesubdirs ignoreversion
Source: "installed.txt"; DestDir: "{app}\app"; Flags: ignoreversion
Source: "install.ps1"; DestDir: "{app}\setup"; Flags: ignoreversion
Source: "requirements-lock.txt"; DestDir: "{app}\setup"; Flags: ignoreversion
Source: "{#Root}\LICENSE"; DestDir: "{app}"; DestName: "LICENSE.txt"; Flags: ignoreversion
Source: "{#Root}\docs\LICENSES.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#Root}\packaging\licenses\*.txt"; DestDir: "{app}\licenses"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\Relight Studio AI"; Filename: "{app}\env\Scripts\pythonw.exe"; Parameters: "-m relight_studio"; WorkingDir: "{app}\app"; IconFilename: "{app}\app\assets\icon.ico"
Name: "{autodesktop}\Relight Studio AI"; Filename: "{app}\env\Scripts\pythonw.exe"; Parameters: "-m relight_studio"; WorkingDir: "{app}\app"; IconFilename: "{app}\app\assets\icon.ico"; Tasks: desktopicon
Name: "{autoprograms}\Repair Relight Studio AI"; Filename: "{sys}\WindowsPowerShell\v1.0\powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\setup\install.ps1"" -Force"; WorkingDir: "{app}"

[Run]
Filename: "{app}\env\Scripts\pythonw.exe"; Parameters: "-m relight_studio"; WorkingDir: "{app}\app"; Description: "{cm:LaunchProgram,Relight Studio AI}"; Flags: nowait postinstall skipifsilent; Check: EnvironmentReady

[UninstallDelete]
; Everything install.ps1 downloaded. Model weights and settings in %LOCALAPPDATA%\RelightStudioAI stay.
Type: filesandordirs; Name: "{app}\env"
Type: filesandordirs; Name: "{app}\python"
Type: filesandordirs; Name: "{app}\tools"
Type: filesandordirs; Name: "{app}\cache"
Type: files; Name: "{app}\install.log"

[Code]
var
  SetupFailed: Boolean;

procedure RemoveTree(const Dir: String);
var
  ResultCode: Integer;
begin
  { The \\?\ prefix lifts the 260-character path limit: package trees nest deeply. }
  if DirExists(Dir) then
    Exec(ExpandConstant('{cmd}'), '/C rmdir /S /Q "\\?\' + Dir + '"', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usUninstall then
  begin
    RemoveTree(ExpandConstant('{app}\env'));
    RemoveTree(ExpandConstant('{app}\python'));
    RemoveTree(ExpandConstant('{app}\tools'));
    RemoveTree(ExpandConstant('{app}\cache'));
  end;
end;

function EnvironmentReady: Boolean;
begin
  Result := (not SetupFailed) and FileExists(ExpandConstant('{app}\env\Scripts\pythonw.exe'));
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  ResultCode: Integer;
  Params: String;
begin
  if CurStep = ssPostInstall then
  begin
    WizardForm.StatusLabel.Caption := 'Downloading Python, PyTorch and the other components...';
    Params := '-NoProfile -ExecutionPolicy Bypass -File "' + ExpandConstant('{app}\setup\install.ps1') + '"';
    if not Exec(ExpandConstant('{sys}\WindowsPowerShell\v1.0\powershell.exe'), Params, ExpandConstant('{app}'),
                SW_SHOW, ewWaitUntilTerminated, ResultCode) or (ResultCode <> 0) then
    begin
      SetupFailed := True;
      SuppressibleMsgBox('The components could not be downloaded or installed. The details are in' + #13#10 +
        ExpandConstant('{app}\install.log') + #13#10#13#10 +
        'Check the internet connection, then run "Repair Relight Studio AI" from the Start menu.',
        mbError, MB_OK, IDOK);
    end;
  end;
end;
