"""
run_dashboard.py
Launcher for the CyberWorld SOC console (local SPAN or Containerlab Lab Mode).

  python run_dashboard.py --site local-default --interface eth1
  python run_dashboard.py --site containerlab-enterprise

Env:
  CYBERWORLD_SITE, CYBERWORLD_SITE_CONFIG, CYBERWORLD_SENSOR_IFACE
"""

import argparse
import os
import sys
from pathlib import Path
import threading
import time
import webbrowser

REPO_ROOT = Path(__file__).resolve().parent
bita_path = str(REPO_ROOT / "bita")

os.chdir(str(REPO_ROOT))
clean_path = [str(REPO_ROOT), bita_path]
for p in (os.environ.get("PYTHONPATH") or "").split(os.pathsep):
    if p and os.path.abspath(p) not in {os.path.abspath(x) for x in clean_path}:
        if os.path.isfile(os.path.join(p, "model.py")):
            continue
        clean_path.append(p)
os.environ["PYTHONPATH"] = os.pathsep.join(clean_path)
sys.path[:0] = [str(REPO_ROOT), bita_path]
for k in list(sys.modules):
    if k == "model" or k.startswith("model."):
        del sys.modules[k]


def open_browser(port: int):
    time.sleep(1.8)
    url = f"http://localhost:{port}"
    print(f"\n[+] Opening SOC Console: {url}\n")
    try:
        webbrowser.open(url)
    except Exception as e:
        print(f"[!] Could not auto-open browser: {e}. Open {url} manually.")


def _parse_args():
    p = argparse.ArgumentParser(description="CyberWorld SOC dashboard")
    p.add_argument("--site", default=None, help="Site id under config/sites/")
    p.add_argument("--site-config", default=None, help="Path to a site YAML file")
    p.add_argument("--interface", default=None, help="Override SPAN capture interface")
    p.add_argument("--replay", default=None, help="Replay PCAP file (demo mode)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--no-browser", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.site_config:
        os.environ["CYBERWORLD_SITE_CONFIG"] = args.site_config
    if args.site:
        os.environ["CYBERWORLD_SITE"] = args.site
    if args.interface:
        os.environ["CYBERWORLD_SENSOR_IFACE"] = args.interface
    if args.replay:
        os.environ["CYBERWORLD_REPLAY_PCAP"] = args.replay

    from control_backend.site_config import get_site_config, reload_site_config

    reload_site_config()
    site = get_site_config()

    print("=" * 70)
    print("  CYBERWORLD SOC — SPAN DISCOVERY + DUAL-BRANCH / DEEPOP")
    print("=" * 70)
    print(f"[+] Repo root:     {REPO_ROOT}")
    print(f"[+] Site:          {site.site_id} (lab_mode={site.lab_mode})")
    print(f"[+] Sensor iface:  {site.sensor_interface}")
    print(f"[+] API & UI:      http://localhost:{args.port}")
    if site.lab_mode:
        print("[+] Mode:          Lab Mode (Containerlab orchestration enabled)")
    else:
        print("[+] Mode:          Local SPAN (no Containerlab required)")
        print("[!] Capture needs CAP_NET_RAW / root on the mirror NIC.")
    print("=" * 70)

    dist_path = REPO_ROOT / "web_dashboard" / "dist" / "index.html"
    if not dist_path.exists():
        print("[!] Frontend dist not found. Building web_dashboard with Vite...")
        os.system(f'cd "{REPO_ROOT / "web_dashboard"}" && npm run build')

    if not args.no_browser:
        threading.Thread(target=open_browser, args=(args.port,), daemon=True).start()

    import uvicorn

    uvicorn.run(
        "control_backend.main:app",
        host=args.host,
        port=args.port,
        reload=False,
    )
