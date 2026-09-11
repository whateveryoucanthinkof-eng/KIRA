"""
web_dashboard/run_dashboard.py
Helper launcher when executed from within web_dashboard folder.
"""
import os
import sys
from pathlib import Path

# Move up to root and execute root launcher
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
os.chdir(str(ROOT_DIR))

from run_dashboard import *

if __name__ == "__main__":
    import uvicorn
    import threading
    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run("control_backend.main:app", host="0.0.0.0", port=8000, reload=False)
