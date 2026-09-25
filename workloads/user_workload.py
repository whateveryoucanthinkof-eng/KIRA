#!/usr/bin/env python3
"""
workloads/user_workload.py
Configurable, realistic normal enterprise employee workload generator.
Generates heterogeneous baseline network traffic across multiple services.
Profiles:
- office: general worker with DNS, web, portal, small files, idle gaps.
- web_heavy: high-frequency DMZ web and external browsing.
- file_heavy: bulk document transfers, spreadsheet downloads, uploads.
- app_heavy: continuous transactional REST API operations against core application.
"""

import json
import os
import random
import socket
import sys
import time
import urllib.request

DNS_SERVER = "10.0.2.10"
ID_SERVER = "http://10.0.2.20:8080"
FILE_SERVER = "http://10.0.2.30:8080"
APP_SERVER = "http://10.0.2.40:8000"
DMZ_WEB = "http://10.0.3.10:80"

DOMAINS = [
    "corp.local", "dns.corp.local", "id.corp.local",
    "file.corp.local", "app.corp.local", "web.corp.local"
]

def send_dns_query(domain="corp.local", server_ip=DNS_SERVER):
    """Generates real UDP DNS query packet to internal DNS server."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(1.5)
        # Construct raw DNS query
        tid = random.randint(1000, 65535).to_bytes(2, "big")
        flags = b"\x01\x00"  # standard query with recursion desired
        qcount = b"\x00\x01"
        zeros = b"\x00\x00\x00\x00\x00\x00"
        
        # Domain name encoding
        qname = b""
        for part in domain.split("."):
            qname += len(part).to_bytes(1, "big") + part.encode("utf-8")
        qname += b"\x00"
        qtype_qclass = b"\x00\x01\x00\x01"  # A record, IN class
        
        packet = tid + flags + qcount + zeros + qname + qtype_qclass
        sock.sendto(packet, (server_ip, 53))
        resp, _ = sock.recvfrom(512)
        sock.close()
        return True
    except Exception:
        return False

def http_get(url):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Workstation; Enterprise-OS)"})
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            return resp.read()
    except Exception:
        return None

def http_post(url, data_bytes, content_type="application/json"):
    try:
        req = urllib.request.Request(
            url,
            data=data_bytes,
            headers={"Content-Type": content_type, "User-Agent": "EnterpriseClient/3.1"}
        )
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            return resp.read()
    except Exception:
        return None

def run_office_profile(node_name):
    """Simulates standard office employee."""
    print(f"[*] Starting Office User Workload for {node_name}...")
    sys.stdout.flush()
    while True:
        action = random.choice(["dns", "web", "app", "file", "idle"])
        if action == "dns":
            send_dns_query(random.choice(DOMAINS))
        elif action == "web":
            http_get(f"{DMZ_WEB}/{random.choice(['', 'about', 'services', 'contact'])}")
        elif action == "app":
            http_get(f"{APP_SERVER}/api/v1/system/status")
        elif action == "file":
            http_get(f"{FILE_SERVER}/{random.choice(['employee_handbook.docx', 'quarterly_earnings.xlsx'])}")
        elif action == "idle":
            time.sleep(random.uniform(2.0, 5.0))
            continue
        time.sleep(random.uniform(1.0, 3.0))

def run_web_heavy_profile(node_name):
    """Simulates user browsing external and DMZ web heavily."""
    print(f"[*] Starting Web-Heavy Workload for {node_name}...")
    sys.stdout.flush()
    paths = ["", "about", "services", "contact"]
    while True:
        send_dns_query("web.corp.local")
        path = random.choice(paths)
        http_get(f"{DMZ_WEB}/{path}")
        if random.random() < 0.2:
            http_post(f"{DMZ_WEB}/contact/submit", json.dumps({"name": "Employee User", "query": "General inquiry"}).encode("utf-8"))
        time.sleep(random.uniform(0.4, 1.2))

def run_file_heavy_profile(node_name):
    """Simulates data-heavy employee uploading and downloading documents."""
    print(f"[*] Starting File-Heavy Workload for {node_name}...")
    sys.stdout.flush()
    files = ["annual_report.pdf", "quarterly_earnings.xlsx", "engineering_design.cad"]
    while True:
        send_dns_query("file.corp.local")
        target_file = random.choice(files)
        http_get(f"{FILE_SERVER}/{target_file}")
        
        # Periodic upload
        if random.random() < 0.4:
            payload = os.urandom(random.randint(4000, 16000))
            http_post(f"{FILE_SERVER}/upload", payload, content_type="application/octet-stream")
        time.sleep(random.uniform(1.5, 3.5))

def run_app_heavy_profile(node_name):
    """Simulates operations staff running continuous enterprise database and backend queries."""
    print(f"[*] Starting Application-Heavy Workload for {node_name}...")
    sys.stdout.flush()
    while True:
        send_dns_query("app.corp.local")
        # Fetch customers or transactions (which triggers backend app -> db connection)
        endpoint = random.choice(["/api/v1/customers", "/api/v1/transactions", "/api/v1/system/status"])
        http_get(f"{APP_SERVER}{endpoint}")
        
        if random.random() < 0.3:
            order_data = json.dumps({"customer_id": 101, "items": [{"item": "ITEM-99", "qty": 2}]}).encode("utf-8")
            http_post(f"{APP_SERVER}/api/v1/orders", order_data)
        time.sleep(random.uniform(0.5, 1.5))

def main():
    profile = sys.argv[1] if len(sys.argv) > 1 else "office"
    node_name = sys.argv[2] if len(sys.argv) > 2 else socket.gethostname()

    # Initial delay to ensure all services are booted
    time.sleep(2.0)

    if profile == "office":
        run_office_profile(node_name)
    elif profile == "web_heavy":
        run_web_heavy_profile(node_name)
    elif profile == "file_heavy":
        run_file_heavy_profile(node_name)
    elif profile == "app_heavy":
        run_app_heavy_profile(node_name)
    else:
        print(f"Unknown profile '{profile}', defaulting to office")
        run_office_profile(node_name)

if __name__ == "__main__":
    main()
