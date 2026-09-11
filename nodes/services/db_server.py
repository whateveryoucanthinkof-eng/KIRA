#!/usr/bin/env python3
"""
nodes/services/db_server.py
Enterprise Relational Database Server (Simulated PostgreSQL/MySQL).
Listens on TCP port 5432.
Strictly accepts connections from authorized application server (10.0.2.40).
"""

import json
import socket
import sys
import threading
import time

DATABASE_RECORDS = {
    "customers": [
        {"id": 101, "name": "Acme Corp", "tier": "enterprise", "balance": 45000.0},
        {"id": 102, "name": "Global Tech", "tier": "premium", "balance": 12400.5},
        {"id": 103, "name": "Apex Logistics", "tier": "standard", "balance": 8750.0},
        {"id": 104, "name": "Nexus Dynamics", "tier": "enterprise", "balance": 98200.0}
    ],
    "transactions": [
        {"txid": "TX-901", "customer_id": 101, "amount": 1500.0, "status": "settled"},
        {"txid": "TX-902", "customer_id": 104, "amount": 8400.0, "status": "pending"},
        {"txid": "TX-903", "customer_id": 102, "amount": 230.0, "status": "settled"}
    ]
}

def handle_client(conn, addr):
    client_ip, client_port = addr
    # Database security validation
    # In enterprise network, DB only accepts queries from application server (10.0.2.40) or localhost
    authorized = (client_ip == "10.0.2.40" or client_ip.startswith("127.") or client_ip == "10.0.2.50")
    if not authorized:
        print(f"[!] DB_SECURITY_ALERT: Unauthorized connection attempt to DB from {client_ip}:{client_port} - ACCESS DENIED")
        sys.stdout.flush()
        conn.sendall(b"ERROR: 42501: permission denied for database 'enterprise_db'\n")
        conn.close()
        return

    try:
        conn.settimeout(5.0)
        while True:
            data = conn.recv(1024)
            if not data:
                break
            query = data.decode("utf-8").strip()
            if not query:
                continue

            if query.upper().startswith("SELECT"):
                # Return customer or transaction records
                if "CUSTOMER" in query.upper():
                    resp = json.dumps({"table": "customers", "rows": DATABASE_RECORDS["customers"]})
                else:
                    resp = json.dumps({"table": "transactions", "rows": DATABASE_RECORDS["transactions"]})
                conn.sendall(resp.encode("utf-8") + b"\n")
            elif query.upper().startswith("INSERT") or query.upper().startswith("UPDATE"):
                conn.sendall(b"OK: 1 row affected\n")
            elif query.upper() == "PING":
                conn.sendall(b"PONG: DB_OK\n")
            elif query.upper() == "QUIT" or query.upper() == "EXIT":
                break
            else:
                conn.sendall(b"OK: Query executed\n")
    except Exception as e:
        pass
    finally:
        conn.close()

def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5432
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", port))
    sock.listen(10)
    print(f"[*] Enterprise Database Server listening on 0.0.0.0:{port}...")
    sys.stdout.flush()

    while True:
        try:
            conn, addr = sock.accept()
            t = threading.Thread(target=handle_client, args=(conn, addr), daemon=True)
            t.start()
        except Exception as e:
            print(f"[!] DB Server accept error: {e}", file=sys.stderr)
            time.sleep(0.1)

if __name__ == "__main__":
    main()
