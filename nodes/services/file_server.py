#!/usr/bin/env python3
"""
nodes/services/file_server.py
Enterprise Storage & Document File Server.
Listens on HTTP port 8080.
Serves and receives simulated corporate documents and financial archives.
"""

import os
import sys
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

FILE_CACHE = {
    "annual_report.pdf": b"%PDF-1.5 Corporate Annual Financial Audit Report " + b"X" * 15000,
    "quarterly_earnings.xlsx": b"PK\x03\x04 Corporate Quarterly Earnings Spreadsheet " + b"Y" * 8000,
    "employee_handbook.docx": b"PK\x03\x04 Enterprise Employee Standards and Handbook " + b"Z" * 5000,
    "engineering_design.cad": b"CAD_V2 Technical Blueprint Schematic Architecture " + b"W" * 25000,
}

class FileServerHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        sys.stdout.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] FILE_SRV: {self.address_string()} - {format%args}\n")
        sys.stdout.flush()

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "healthy", "service": "file-server"}\n')
            return

        filename = self.path.lstrip("/").split("?")[0]
        if filename in FILE_CACHE:
            content = FILE_CACHE[filename]
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        else:
            # Return file directory listing
            files_html = "<html><body><h1>Enterprise Shared Files</h1><ul>"
            for f in FILE_CACHE:
                files_html += f'<li><a href="/{f}">{f}</a> ({len(FILE_CACHE[f])} bytes)</li>'
            files_html += "</ul></body></html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(files_html.encode("utf-8"))

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        data = self.rfile.read(content_length) if content_length > 0 else b""
        upload_name = f"upload_{int(time.time())}.dat"
        FILE_CACHE[upload_name] = data
        self.send_response(201)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(f'{{"status": "saved", "file": "{upload_name}", "bytes": {len(data)}}}\n'.encode("utf-8"))

def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    server = HTTPServer(("0.0.0.0", port), FileServerHandler)
    print(f"[*] Enterprise File Server listening on 0.0.0.0:{port}...")
    sys.stdout.flush()
    server.serve_forever()

if __name__ == "__main__":
    main()
