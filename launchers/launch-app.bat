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

REM Convert the folder holding the .sif to a WSL path for portability
for /f "delims=" %%i in ('wsl wslpath -a "%SIFDIR%"') do set WSLPATH=%%i

REM Launch Voila in a new WSL window (stays open so you can see logs)
start "QuantUI" wsl -d Ubuntu -- bash -c "cd '%WSLPATH%' && apptainer run quantui.sif app"

REM Wait for Voila to start, then open the browser
echo Waiting for Voila to start...
timeout /t 6 /nobreak > nul
start http://localhost:8866

echo.
echo App is running at http://localhost:8866
echo Close the WSL window to stop the server.
