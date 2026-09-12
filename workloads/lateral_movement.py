#!/usr/bin/env python3
import time
import sys
import urllib.request
import urllib.parse
import json

def trigger_fake_dashboard():
    try:
        url = "http://localhost:8001/api/trigger?attack_type=LateralMovement"
        req = urllib.request.Request(url, method="POST")
        with urllib.request.urlopen(req, timeout=2) as response:
            pass
    except Exception:
        pass

def cancel_fake_dashboard():
    try:
        url = "http://localhost:8001/api/cancel"
        req = urllib.request.Request(url, method="POST")
        with urllib.request.urlopen(req, timeout=2) as response:
            pass
    except Exception:
        pass

def main():
    print("[*] Loading Pass-the-Hash module...")
    time.sleep(1)
    print("[*] Target internal subnet: 10.0.2.0/24")
    time.sleep(1)
    
    # Trigger the dashboard visualization
    trigger_fake_dashboard()
    
    print("[+] Exploiting SMB vulnerabilities...")
    print("[*] Attempting lateral movement pivot (Press Ctrl+C to cancel)...")
    
    try:
        for i in range(10, 20):
            print(f"[+] Successfully authenticated to 10.0.2.{i}")
            time.sleep(1.5)
        print("\n[*] Lateral movement completed.")
        cancel_fake_dashboard()
    except KeyboardInterrupt:
        print("\n[*] Module interrupted.")
        cancel_fake_dashboard()
        print("[+] Attack cancelled. Dashboard returning to baseline.")

if __name__ == "__main__":
    main()
