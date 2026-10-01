"""
web_dashboard/run_dashboard.py
Convenience launcher when run from inside web_dashboard/.

Hands every argument to the repository-root run_dashboard.py, so there is one
launcher with one security posture (loopback by default, access token when
exposed):

    python run_dashboard.py            # the real console
    python run_dashboard.py --demo     # the demo dashboard (sample data)
"""
import os
import sys
from pathlib import Path

ROOT_LAUNCHER = Path(__file__).resolve().parent.parent / "run_dashboard.py"

if __name__ == "__main__":
    os.execv(sys.executable, [sys.executable, str(ROOT_LAUNCHER), *sys.argv[1:]])
