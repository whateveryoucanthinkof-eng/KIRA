@echo off
cd /d "%~dp0\.."
echo Starting CyberWorld SOC Console from repository root...
python run_dashboard.py
pause
