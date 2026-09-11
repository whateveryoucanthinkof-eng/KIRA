#!/usr/bin/env python3
"""
nodes/services/web_dmz.py
DMZ Public Enterprise Web Server.
Listens on HTTP port 80.
Serves public corporate website, contact portal, and external services.
"""

import json
import sys
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

HTML_HOME = """<!DOCTYPE html>
<html>
<head><title>Nexus Enterprise Solutions</title></head>
<body style="font-family: Arial, sans-serif; margin: 40px;">
  <h1>Nexus Enterprise Solutions — Global Portal</h1>
  <p>Securing next-generation autonomous enterprise operations.</p>
  <nav>
    <a href="/">Home</a> | <a href="/about">About Us</a> | <a href="/services">Services</a> | <a href="/contact">Contact</a>
  </nav>
  <hr>
  <h2>Operational Status</h2>
  <p>All perimeter services online and nominal.</p>
</body>
</html>
"""

class DMZWebHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        sys.stdout.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] DMZ_WEB: {self.address_string()} - {format%args}\n")
        sys.stdout.flush()

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "healthy", "service": "dmz-web", "zone": "dmz"}\n')

        elif self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(HTML_HOME.encode("utf-8"))

        elif self.path == "/about":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<html><body><h1>About Nexus</h1><p>Global leader in critical infrastructure technology.</p></body></html>\n")

        elif self.path == "/services":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            services = [
                {"service": "cloud_storage", "sla": "99.99%"},
                {"service": "threat_intelligence", "sla": "99.95%"},
                {"service": "managed_dns", "sla": "100.0%"}
            ]
            self.wfile.write(json.dumps(services).encode("utf-8") + b"\n")

        elif self.path == "/contact":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<html><body><h1>Contact Operations</h1><p>Email: contact@corp.local</p></body></html>\n")

        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"404 Not Found\n")

    def do_POST(self):
        if self.path == "/contact/submit":
            content_length = int(self.headers.get("Content-Length", 0))
            data = self.rfile.read(content_length)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "received", "ticket": "TKT-8841"}\n')
        else:
            self.send_response(404)
            self.end_headers()

def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    server = HTTPServer(("0.0.0.0", port), DMZWebHandler)
    print(f"[*] DMZ Web Server listening on 0.0.0.0:{port}...")
    sys.stdout.flush()
    server.serve_forever()

if __name__ == "__main__":
    main()
