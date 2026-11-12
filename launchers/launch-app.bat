@echo off
echo QuantUI — Starting...
echo.

REM Locate quantui.sif — next to this launcher (student download) or in the
REM repo root one level up (dev build via apptainer/build.sh).
set "SIFDIR=%~dp0"
if not exist "%SIFDIR%quantui.sif" set "SIFDIR=%~dp0..\"
if not exist "%SIFDIR%quantui.sif" (
    echo ERROR: quantui.sif not found.
    echo Build it first:  bash apptainer/build.sh
    echo Or download it from the GitHub Releases page.
    pause
    exit /b 1
)

REM Convert the folder holding the .sif to a WSL path for portability. Not
REM "%SIFDIR%" itself: it ends in "\", which escapes the closing quote.
for %%i in ("%SIFDIR%.") do set "SIFDIR=%%~fi"
for /f "delims=" %%i in ('wsl wslpath -a "%SIFDIR%"') do set WSLPATH=%%i

REM Something already serving on 8866 (an earlier run, another app) would be
REM opened in the browser instead of this one, so refuse to start.
powershell -NoProfile -Command "$c = New-Object Net.Sockets.TcpClient; try { $c.Connect('127.0.0.1', 8866); $c.Close(); exit 0 } catch { exit 1 }" > nul 2>&1
if not errorlevel 1 (
    echo ERROR: port 8866 is already in use - is QuantUI already running?
    echo Close that window or program first, then run this again.
    pause
    exit /b 1
)

REM Per-run login token: without one, any other local user or process could
REM drive the app. WSLENV hands QUANTUI_APP_TOKEN to the WSL side through the
REM environment, not a command line.
set "QUANTUI_APP_TOKEN="
for /f "delims=" %%t in ('powershell -NoProfile -Command "[guid]::NewGuid().Guid + [guid]::NewGuid().Guid"') do set "QUANTUI_APP_TOKEN=%%t"
if not defined QUANTUI_APP_TOKEN (
    echo ERROR: could not create a login token ^(is PowerShell available?^)
    pause
    exit /b 1
)
if defined WSLENV (set "WSLENV=%WSLENV%:QUANTUI_APP_TOKEN") else (set "WSLENV=QUANTUI_APP_TOKEN")

REM Launch Voila in a new WSL window (stays open so you can see logs). --cd
REM rather than a "cd '...'" inside bash -c, which broke on an apostrophe in
REM the folder path.
start "QuantUI" wsl -d Ubuntu --cd "%WSLPATH%" -- bash -c "apptainer run quantui.sif app"

REM Wait (up to 90 s) for Voila to answer on the port, then open the browser
echo Waiting for Voila to start...
powershell -NoProfile -Command "for ($i = 0; $i -lt 90; $i++) { $c = New-Object Net.Sockets.TcpClient; try { $c.Connect('127.0.0.1', 8866); $c.Close(); exit 0 } catch { Start-Sleep -Seconds 1 } }; exit 1" > nul 2>&1
if errorlevel 1 echo Voila did not answer within 90 s - check the QuantUI WSL window.
start http://localhost:8866/?token=%QUANTUI_APP_TOKEN%

echo.
echo App is running at http://localhost:8866/?token=%QUANTUI_APP_TOKEN%
echo Close the WSL window to stop the server.
