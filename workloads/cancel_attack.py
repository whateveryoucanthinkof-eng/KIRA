#!/usr/bin/env python3
"""
workloads/cancel_attack.py
Disarms and cancels the active attack simulation on the fake dashboard,
bringing both the observed and predicted graphs smoothly back down to baseline.
"""

import sys
import time
import urllib.request

def cancel_attack():
    url = "http://localhost:8001/api/cancel"
    print("[*] Contacting CyberWorld SOC Fake Dashboard on port 8001...")
    try:
        req = urllib.request.Request(url, method="POST")
        with urllib.request.urlopen(req, timeout=3) as response:
            if response.status == 200:
                print("[+] Attack successfully CANCELLED.")
                print("[+] Telemetry and prediction lines returning to baseline.")
                print("[+] Threat status: Disarmed (LOW).")
                return True
    except Exception as e:
        print(f"[!] Could not reach fake dashboard at {url}: {e}")
        print("[!] Make sure ./run_fake_dashboard.py is running on port 8001.")
        return False

if __name__ == "__main__":
    cancel_attack()
