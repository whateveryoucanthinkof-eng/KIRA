#!/usr/bin/env python3
"""
run_fake_dashboard.py
Launcher for the CyberWorld SOC Fake Demo Dashboard.
This runs a completely simulated version of the dashboard for demonstration purposes.
"""

import argparse
import os
import sys
from pathlib import Path
import threading
import time
import webbrowser
import uvicorn

def open_browser(port: int):
    time.sleep(1.8)
    url = f"http://localhost:{port}"
    print(f"\n[+] Opening Fake SOC Console: {url}\n")
    try:
        webbrowser.open(url)
    except Exception as e:
        print(f"[!] Could not auto-open browser: {e}. Open {url} manually.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fake CyberWorld SOC dashboard")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("  CYBERWORLD SOC — FAKE DEMO DASHBOARD")
    print("=" * 70)
    print(f"[+] API & UI:      http://localhost:{args.port}")
    print("=" * 70)

    if not args.no_browser:
        threading.Thread(target=open_browser, args=(args.port,), daemon=True).start()

    uvicorn.run(
        "fake_backend:app",
        host=args.host,
        port=args.port,
        reload=False,
    )
