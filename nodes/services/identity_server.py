#!/usr/bin/env python3
"""
nodes/services/identity_server.py
Enterprise Identity & Authentication Service (LDAP/OAuth mock).
Listens on HTTP port 8080.
"""

import json
import sys
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

USERS = {
    "user_office": {"role": "employee", "dept": "administration", "clearance": "standard"},
    "user_web": {"role": "marketing", "dept": "communications", "clearance": "standard"},
    "user_file": {"role": "finance", "dept": "accounting", "clearance": "confidential"},
    "user_app": {"role": "engineer", "dept": "operations", "clearance": "restricted"},
    "admin": {"role": "sysadmin", "dept": "it_sec", "clearance": "top_secret"}
}

class IdentityHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Concise logging
        sys.stdout.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ID_SRV: {self.address_string()} - {format%args}\n")
        sys.stdout.flush()

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "healthy", "service": "identity-server"}\n')
        elif self.path.startswith("/users/"):
            username = self.path.split("/")[-1]
            if username in USERS:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(USERS[username]).encode("utf-8") + b"\n")
            else:
                self.send_response(404)
                self.end_headers()
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Enterprise Identity Service Active\n")

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b"{}"
        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception:
            payload = {}

        if self.path == "/auth/token":
            username = payload.get("username", "user_office")
            user_info = USERS.get(username, {"role": "guest", "clearance": "unclassified"})
            token = f"jwt-corp-{username}-{int(time.time())}"
            resp = {
                "token": token,
                "expires_in": 3600,
                "user": username,
                "profile": user_info
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode("utf-8") + b"\n")

        elif self.path == "/auth/verify":
            token = payload.get("token", "")
            valid = token.startswith("jwt-corp-")
            self.send_response(200 if valid else 401)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"valid": valid, "timestamp": time.time()}).encode("utf-8") + b"\n")
        else:
            self.send_response(404)
            self.end_headers()

def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    server = HTTPServer(("0.0.0.0", port), IdentityHandler)
    print(f"[*] Enterprise Identity Server listening on 0.0.0.0:{port}...")
    sys.stdout.flush()
    server.serve_forever()

if __name__ == "__main__":
    main()
