#!/usr/bin/env python3
import time
import sys
import urllib.request
import urllib.parse
import json

def trigger_fake_dashboard():
    try:
        url = "http://localhost:8001/api/trigger?attack_type=DDoS"
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
    print("[*] Initializing high-volume UDP flood...")
    time.sleep(1)
    print("[*] Target: 10.0.3.10 (DMZ Web Server)")
    time.sleep(1)
    
    # Trigger the dashboard visualization (predicted line spikes first!)
    trigger_fake_dashboard()
    
    print("[+] Attack launched successfully.")
    print("[*] Generating packet flood across port 80/443 (Press Ctrl+C to cancel/stop)...")
    
    try:
        while True:
            sys.stdout.write(".")
            sys.stdout.flush()
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[*] Stopping attack...")
        cancel_fake_dashboard()
        print("[+] Attack cancelled. Dashboard returning to baseline.")

if __name__ == "__main__":
    main()
