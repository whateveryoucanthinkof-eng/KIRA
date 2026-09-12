#!/usr/bin/env python3
import time
import sys
import urllib.request
import urllib.parse
import json

def trigger_fake_dashboard():
    try:
        url = "http://localhost:8001/api/trigger?attack_type=C2Beaconing"
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
    print("[*] Initializing Covert C2 Channel...")
    time.sleep(1)
    print("[*] Target endpoint: 192.168.100.10:443")
    time.sleep(1)
    
    # Trigger the dashboard visualization
    trigger_fake_dashboard()
    
    print("[+] Establishing encrypted tunnel via HTTPS...")
    print("[*] Sending keep-alive beacons (Press Ctrl+C to stop)...")
    
    try:
        while True:
            print("[+] Beacon sent. (0 bytes data)")
            time.sleep(2)
    except KeyboardInterrupt:
        print("\n[*] C2 connection closed.")
        cancel_fake_dashboard()
        print("[+] Attack cancelled. Dashboard returning to baseline.")

if __name__ == "__main__":
    main()
