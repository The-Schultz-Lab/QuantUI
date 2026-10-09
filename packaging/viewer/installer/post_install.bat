@echo off
rem Install the bundled QuantUI wheel, then add Desktop + Start-menu shortcuts
rem that run "quantui view" (the console window is the server; closing it or
rem pressing Exit in the app stops QuantUI). Output: %PREFIX%\post_install.log
set "LOG=%PREFIX%\post_install.log"
for %%f in ("%PREFIX%\quantui-*.whl") do (
    "%PREFIX%\python.exe" -m pip install --no-deps --no-index --no-cache-dir "%%f" >> "%LOG%" 2>&1 || exit /b 1
    del "%%f"
)
"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File "%PREFIX%\create_shortcuts.ps1" -Prefix "%PREFIX%" >> "%LOG%" 2>&1
rem A missing shortcut should not fail the install; the log records why.
exit /b 0
