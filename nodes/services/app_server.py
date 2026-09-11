#!/usr/bin/env python3
"""
nodes/services/app_server.py
Enterprise Core Application Server.
Listens on HTTP port 8000.
Interacts with Identity Server (10.0.2.20:8080) and Database Server (10.0.2.50:5432).
"""

import json
import socket
import sys
import time
import urllib.request
from http.server import HTTPServer, BaseHTTPRequestHandler

ID_SERVER_URL = "http://10.0.2.20:8080"
DB_SERVER_HOST = "10.0.2.50"
DB_SERVER_PORT = 5432

def query_database(sql_query: str) -> str:
    """Connects to srv-db via TCP port 5432 and executes a query."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2.0)
        s.connect((DB_SERVER_HOST, DB_SERVER_PORT))
        s.sendall(sql_query.encode("utf-8") + b"\n")
        data = s.recv(4096)
        s.close()
        return data.decode("utf-8").strip()
    except Exception as e:
        return json.dumps({"error": f"Database unreachable: {e}"})

def verify_token_with_id_service(token: str) -> bool:
    """Verifies token against srv-id."""
    try:
        req = urllib.request.Request(
            f"{ID_SERVER_URL}/auth/verify",
            data=json.dumps({"token": token}).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("valid", False)
    except Exception:
        # Fallback if id server is slow/offline
        return token.startswith("jwt-corp-")

class AppServerHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        sys.stdout.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] APP_SRV: {self.address_string()} - {format%args}\n")
        sys.stdout.flush()

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "healthy", "service": "app-server", "tier": "business_logic"}\n')

        elif self.path.startswith("/api/v1/customers"):
            # Multi-tier transaction: queries database
            db_res = query_database("SELECT * FROM customers")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(db_res.encode("utf-8") + b"\n")

        elif self.path.startswith("/api/v1/transactions"):
            db_res = query_database("SELECT * FROM transactions")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(db_res.encode("utf-8") + b"\n")

        elif self.path == "/api/v1/system/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            status = {
                "uptime": time.time(),
                "active_connections": 12,
                "db_backend": f"{DB_SERVER_HOST}:{DB_SERVER_PORT}",
                "auth_backend": ID_SERVER_URL
            }
            self.wfile.write(json.dumps(status).encode("utf-8") + b"\n")
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Enterprise Core Application API v1.0\n")

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b"{}"
        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception:
            payload = {}

        if self.path == "/api/v1/orders":
            # Writes transaction to DB
            query_database("INSERT INTO transactions VALUES ('TX-NEW', 101, 500.0, 'settled')")
            self.send_response(201)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "created", "order_id": "ORD-5542"}\n')
        else:
            self.send_response(404)
            self.end_headers()

def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    server = HTTPServer(("0.0.0.0", port), AppServerHandler)
    print(f"[*] Enterprise Application Server listening on 0.0.0.0:{port}...")
    sys.stdout.flush()
    server.serve_forever()

if __name__ == "__main__":
    main()
