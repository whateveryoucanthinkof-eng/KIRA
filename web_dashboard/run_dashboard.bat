@echo off
cd /d "%~dp0\.."
echo Starting CyberFortress SOC Console from repository root...
python run_dashboard.py
pause
