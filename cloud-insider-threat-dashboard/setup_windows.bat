@echo off
cd /d "%~dp0"
py -3.12 -m venv .venv
if errorlevel 1 goto failed
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 goto failed
echo.
echo Setup complete. Double-click run_app.bat to start the dashboard.
pause
exit /b 0
:failed
echo.
echo Setup failed. Install Python 3.12 and try again.
pause
exit /b 1
