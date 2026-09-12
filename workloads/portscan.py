#!/usr/bin/env python3
import time
import sys
import urllib.request
import urllib.parse
import json

def trigger_fake_dashboard():
    try:
        url = "http://localhost:8001/api/trigger?attack_type=PortScan"
        req = urllib.request.Request(url, method="POST")
        with urllib.request.urlopen(req) as response:
            pass
    except Exception:
        pass

def main():
    print("[*] Starting Nmap stealth SYN scan (nmap -sS -p 1-65535)...")
    time.sleep(1)
    print("[*] Target: 10.0.3.0/24")
    time.sleep(1)
    
    # Trigger the dashboard visualization
    trigger_fake_dashboard()
    
    print("[+] Scan initiated.")
    print("[*] Discovering open ports...")
    
    try:
        for i in range(1, 100):
            sys.stdout.write(f"Scanned {i*650} ports...\r")
            sys.stdout.flush()
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\n[*] Scan aborted.")

if __name__ == "__main__":
    main()
