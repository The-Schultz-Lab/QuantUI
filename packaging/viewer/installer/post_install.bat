@echo off
rem Install the bundled QuantUI wheel, then add Desktop + Start-menu shortcuts
rem that run "quantui view" (the console window is the server; closing it or
rem pressing Exit in the app stops QuantUI).
for %%f in ("%PREFIX%\quantui-*.whl") do (
    "%PREFIX%\python.exe" -m pip install --no-deps --no-index --no-cache-dir "%%f" || exit /b 1
    del "%%f"
)
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$s = New-Object -ComObject WScript.Shell;" ^
  "foreach ($d in @([Environment]::GetFolderPath('Desktop'), [Environment]::GetFolderPath('Programs'))) {" ^
  "  $l = $s.CreateShortcut((Join-Path $d 'QuantUI Viewer.lnk'));" ^
  "  $l.TargetPath = '%PREFIX%\Scripts\quantui.exe';" ^
  "  $l.Arguments = 'view';" ^
  "  $l.WorkingDirectory = $env:USERPROFILE;" ^
  "  $l.Save() }"
exit /b 0
