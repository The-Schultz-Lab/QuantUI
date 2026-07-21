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

REM Uses the local notebook on disk instead of the baked-in copy.
REM Edits to notebooks/ take effect immediately — no container rebuild needed.
start "QuantUI [dev]" wsl -d Ubuntu -- bash -c "cd '%WSLPATH%' && apptainer exec --cleanenv quantui.sif voila notebooks/molecule_computations.ipynb --no-browser --port=8866 --ServerApp.disable_check_xsrf=True"

echo Waiting for Voila to start...
timeout /t 6 /nobreak > nul
start http://localhost:8866

echo.
echo Dev server running at http://localhost:8866
echo Edit notebooks/ and refresh the browser — no rebuild needed.
echo Close the WSL window to stop.
