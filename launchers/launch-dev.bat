@echo off
echo QuantUI DEV MODE — Using local notebook (no rebuild needed)
echo.

REM Repo root is one level up from this launchers/ folder.
for %%i in ("%~dp0..") do set "REPO=%%~fi"

if not exist "%REPO%\quantui.sif" (
    echo ERROR: quantui.sif not found.
    pause
    exit /b 1
)

echo NOTE: The container provides the quantui Python package.
echo       Notebook edits (notebooks/) are always live.
echo       Edits to quantui/*.py require a container rebuild to take effect.
echo       For package-level dev, use launch-native.bat instead.
echo.

REM Convert the repo root to a WSL path for portability
for /f "delims=" %%i in ('wsl wslpath -a "%REPO%"') do set WSLPATH=%%i

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

REM Something already serving on 8866 (an earlier run, another app) would be
REM opened in the browser instead of this one, so refuse to start.
powershell -NoProfile -Command "$c = New-Object Net.Sockets.TcpClient; try { $c.Connect('127.0.0.1', 8866); $c.Close(); exit 0 } catch { exit 1 }" > nul 2>&1
if not errorlevel 1 (
    echo ERROR: port 8866 is already in use - is QuantUI already running?
    echo Close that window or program first, then run this again.
    pause
    exit /b 1
)

REM Uses the local notebook on disk instead of the baked-in copy.
REM Edits to notebooks/ take effect immediately — no container rebuild needed.
start "QuantUI [dev]" wsl -d Ubuntu --cd "%WSLPATH%" -- bash -c "apptainer exec --cleanenv quantui.sif voila notebooks/molecule_computations.ipynb --no-browser --port=8866 --Voila.ip=127.0.0.1 --Voila.token=$QUANTUI_APP_TOKEN --ServerApp.disable_check_xsrf=True"

REM The WSL window starts Voila with --cd rather than a "cd '...'" inside bash -c,
REM which broke on an apostrophe in the folder path. Wait (up to 90 s) for it
REM to answer on the port, then open the browser.
echo Waiting for Voila to start...
powershell -NoProfile -Command "for ($i = 0; $i -lt 90; $i++) { $c = New-Object Net.Sockets.TcpClient; try { $c.Connect('127.0.0.1', 8866); $c.Close(); exit 0 } catch { Start-Sleep -Seconds 1 } }; exit 1" > nul 2>&1
if errorlevel 1 echo Voila did not answer within 90 s - check the QuantUI [dev] WSL window.
start http://localhost:8866/?token=%QUANTUI_APP_TOKEN%

echo.
echo Dev server running at http://localhost:8866/?token=%QUANTUI_APP_TOKEN%
echo Edit notebooks/ and refresh the browser — no rebuild needed.
echo Close the WSL window to stop.
