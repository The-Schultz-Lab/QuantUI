@echo off
echo QuantUI NATIVE JUPYTER MODE -- Local conda env in WSL, no container
echo Use this when you have edited quantui/*.py and want JupyterLab.
echo.

REM Repo root is one level up from this launchers/ folder; convert it to a WSL
REM path for portability.
for %%i in ("%~dp0..") do set "REPO=%%~fi"
for /f "delims=" %%i in ('wsl wslpath -a "%REPO%"') do set WSLPATH=%%i
if not defined WSLPATH (
	echo ERROR: Could not resolve a WSL path for %REPO%
	echo Try this command manually:
	echo   wsl wslpath -a "%REPO%"
	echo.
	pause
	exit /b 1
)
set "LOGFILE=%REPO%\logs\native-jupyter.log"

echo Startup log: %LOGFILE%
echo.

REM Runs JupyterLab directly from the quantui conda env inside WSL.
REM pip install -e . is skipped when pyproject.toml has not changed since the
REM last install (.dev_install_stamp). quantui/*.py changes are always live in
REM editable mode -- reinstall is only needed after pyproject.toml changes or on
REM first use.
REM Uses port 8868 to avoid conflict with container-based launchers on 8866 and
REM native Voila launcher on 8867.
REM Clears quantui/__pycache__ on every launch to prevent stale .pyc bytecode
REM (WSL2 DrvFs does not reliably propagate Windows-side mtime changes, so Python
REM may load pre-edit bytecode even after source changes -- see GOTCHAS.md).
REM PYTHONDONTWRITEBYTECODE=1 prevents a new stale cache from accumulating.
start "QuantUI [native-jupyter]" wsl -d Ubuntu --cd "%WSLPATH%" -- bash ./launchers/launch-native-jupyter.sh

echo Waiting for JupyterLab to start on localhost:8868...
set MAX_WAIT=45
set waited=0
set OPENED=0

:wait_for_jupyter
powershell -NoProfile -Command "$client = New-Object Net.Sockets.TcpClient; try { $client.Connect('127.0.0.1', 8868); $client.Close(); exit 0 } catch { exit 1 }" > nul 2>&1
if %errorlevel%==0 goto open_browser
if %waited% GEQ %MAX_WAIT% goto startup_timeout
timeout /t 1 /nobreak > nul
set /a waited=%waited%+1
goto wait_for_jupyter

:startup_timeout
echo.
echo JupyterLab did not open localhost:8868 within %MAX_WAIT% seconds.
echo Check the QuantUI [native-jupyter] WSL window for startup errors.
echo Review startup log: %LOGFILE%
if exist "%LOGFILE%" start "" "%LOGFILE%"
echo.
goto done

:open_browser
set OPENED=1
start http://127.0.0.1:8868/lab/tree/notebooks/molecule_computations.ipynb

:done

echo.
if "%OPENED%"=="1" (
	echo Native JupyterLab server running at http://127.0.0.1:8868/lab
	echo All local quantui/*.py changes are live -- no rebuild needed.
	echo Close the WSL window to stop.
) else (
	echo JupyterLab startup not confirmed yet.
	echo Review the QuantUI [native-jupyter] WSL window for details.
	echo Startup log: %LOGFILE%
)
