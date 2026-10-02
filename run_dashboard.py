"""
run_dashboard.py
Launcher for the K.I.R.A. SOC console.

The real console (recommended) -- FastAPI backend + live models + the UI:

  python run_dashboard.py                                    # site from CYBERWORLD_SITE
  python run_dashboard.py --site containerlab-enterprise     # Containerlab Lab Mode
  python run_dashboard.py --site local-default --interface eth1
  python run_dashboard.py --replay captures/live.pcap        # replay a capture

The demo dashboard -- only for seeing what the UI looks like. Sample data, no
backend, no sensor, no models; nothing it shows is a model output:

  python run_dashboard.py --demo

The UI is built on demand (web_dashboard/dist for the console, dist-demo for
the demo) whenever its sources are newer than the build. Building needs npm on
PATH, or inside a toolbox named in CYBERWORLD_NPM_TOOLBOX (optional).

Env:
  CYBERWORLD_SITE, CYBERWORLD_SITE_CONFIG, CYBERWORLD_SENSOR_IFACE
  CYBERWORLD_API_TOKEN   require this token on every request (auto-generated
                         when --host is not a loopback address)
  CYBERWORLD_NPM_TOOLBOX toolbox to run npm in when npm is not on PATH

Binds 127.0.0.1 by default. The console shows the internal host map and can
record mitigations, so exposing it on a shared network needs a token.
"""

import argparse
import ipaddress
import os
import secrets
import shutil
import subprocess
import sys
from pathlib import Path
import threading
import time
import webbrowser

REPO_ROOT = Path(__file__).resolve().parent
bita_path = str(REPO_ROOT / "bita")

# Why this path juggling exists: the vendored TGNE-TA code under bita/ imports its own
# submodules by bare top-level names -- `from model.tgn import TGN`, `from modules.memory
# import Memory`, `from utils.utils import ...`. That only resolves with bita/ itself on
# sys.path, which puts a generic `model` package into the global namespace. Any other
# `model.py` on PYTHONPATH (or a `model` already imported by something else) shadows it and
# breaks checkpoint loading, so entries owning a model.py are dropped and a stale `model`
# module is evicted below. Repackaging bita/ into a proper namespace would remove all of
# this; it is left alone deliberately because the checkpoints are keyed to that layout.
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


def _is_loopback(host: str) -> bool:
    if host in ("localhost",):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def open_browser(port: int):
    time.sleep(1.8)
    url = f"http://localhost:{port}"
    token = os.environ.get("CYBERWORLD_API_TOKEN", "")
    if token:
        url += f"/?token={token}"
    print(f"\n[+] Opening SOC Console: {url}\n")
    try:
        webbrowser.open(url)
    except Exception as e:
        print(f"[!] Could not auto-open browser: {e}. Open {url} manually.")


WEB_DIR = REPO_ROOT / "web_dashboard"


def _newest_source_mtime() -> float:
    """Latest change to anything the UI build reads."""
    newest = 0.0
    for p in [WEB_DIR / "index.html", WEB_DIR / "vite.config.ts", WEB_DIR / "package.json", WEB_DIR / ".env.demo"]:
        if p.exists():
            newest = max(newest, p.stat().st_mtime)
    for root in (WEB_DIR / "src", WEB_DIR / "public"):
        for dirpath, _dirs, files in os.walk(root):
            for f in files:
                newest = max(newest, os.stat(os.path.join(dirpath, f)).st_mtime)
    return newest


def _npm_command():
    """npm on PATH, else npm inside a toolbox; None when neither is available."""
    if shutil.which("npm"):
        return ["npm"]
    box = os.environ.get("CYBERWORLD_NPM_TOOLBOX", "")
    if box and shutil.which("toolbox"):
        probe = subprocess.run(["toolbox", "run", "-c", box, "npm", "--version"],
                               capture_output=True, text=True)
        if probe.returncode == 0:
            return ["toolbox", "run", "-c", box, "npm"]
    return None


def ensure_frontend(demo: bool, force: bool = False) -> Path:
    """Build the UI when its build is missing or older than its sources."""
    out = WEB_DIR / ("dist-demo" if demo else "dist")
    index = out / "index.html"
    stale = force or not index.exists() or index.stat().st_mtime < _newest_source_mtime()
    if not stale:
        return out
    npm = _npm_command()
    if npm is None:
        if index.exists():
            print(f"[!] {out.name}/ is older than the UI sources, but npm is not available "
                  "to rebuild it. Serving the existing build.")
            return out
        sys.exit(
            f"[x] The UI has not been built ({out.relative_to(REPO_ROOT)} is missing) and npm is not "
            "available.\n    Install Node.js 20+, or set CYBERWORLD_NPM_TOOLBOX to a toolbox that has "
            "npm, then run this again."
        )
    print(f"[*] Building the {'demo dashboard' if demo else 'console'} UI ({' '.join(npm)})...")
    if not (WEB_DIR / "node_modules").exists():
        subprocess.run([*npm, "ci", "--no-audit", "--no-fund"], cwd=WEB_DIR, check=True)
    subprocess.run([*npm, "run", "build:demo" if demo else "build"], cwd=WEB_DIR, check=True)
    return out


def serve_demo(port: int, open_in_browser: bool, force_build: bool) -> None:
    """Serve the demo dashboard: a static bundle on sample data, no backend."""
    import functools
    import http.server

    out = ensure_frontend(demo=True, force=force_build)

    class SpaHandler(http.server.SimpleHTTPRequestHandler):
        # Unknown paths fall back to index.html, as the console is one page.
        def send_head(self):
            path = self.translate_path(self.path)
            if not os.path.exists(path):
                self.path = "/index.html"
            return super().send_head()

        def log_message(self, *args):
            pass

    print("=" * 70)
    print("  K.I.R.A. DEMO DASHBOARD  --  sample data only")
    print("=" * 70)
    print("[!] Nothing on this page is a model output. It exists to show the UI.")
    print("[i] For the real console run:  python run_dashboard.py")
    print(f"[+] Demo UI: http://localhost:{port}")
    print("=" * 70)
    handler = functools.partial(SpaHandler, directory=str(out))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    if open_in_browser:
        threading.Thread(target=open_browser, args=(port,), daemon=True).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


def _parse_args():
    p = argparse.ArgumentParser(description="K.I.R.A. SOC console")
    p.add_argument("--demo", action="store_true",
                   help="Demo dashboard: the UI on sample data, no backend or models. "
                        "Only for seeing how the console looks.")
    p.add_argument("--rebuild", action="store_true", help="Rebuild the UI even if it looks current")
    p.add_argument("--site", default=None, help="Site id under config/sites/")
    p.add_argument("--site-config", default=None, help="Path to a site YAML file")
    p.add_argument("--interface", default=None, help="Override SPAN capture interface")
    p.add_argument("--replay", default=None, help="Replay PCAP file (demo mode)")
    p.add_argument("--host", default="127.0.0.1",
                   help="Bind address. Default loopback only; any other address "
                        "requires an access token (generated if not set).")
    p.add_argument("--port", type=int, default=None,
                   help="Port (default 8000 for the console, 8443 for --demo)")
    p.add_argument("--no-browser", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.demo:
        serve_demo(args.port or 8443, not args.no_browser, args.rebuild)
        sys.exit(0)
    args.port = args.port or 8000
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
    print("  K.I.R.A. SOC CONSOLE — SPAN DISCOVERY + DUAL-BRANCH / DEEPOP")
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

    if not _is_loopback(args.host) and not os.environ.get("CYBERWORLD_API_TOKEN"):
        os.environ["CYBERWORLD_API_TOKEN"] = secrets.token_urlsafe(24)
    token = os.environ.get("CYBERWORLD_API_TOKEN", "")
    if token:
        print(f"[+] Access token required. Open: http://localhost:{args.port}/?token={token}")
    if not _is_loopback(args.host):
        print(f"[!] Binding {args.host}: reachable from other machines on this network.")

    ensure_frontend(demo=False, force=args.rebuild)

    if not args.no_browser:
        threading.Thread(target=open_browser, args=(args.port,), daemon=True).start()

    import uvicorn

    uvicorn.run(
        "control_backend.main:app",
        host=args.host,
        port=args.port,
        reload=False,
    )
