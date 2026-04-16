#!/usr/bin/env python
"""Container health check script.

Polls the /health endpoint of the running Flask app and exits 0 on success,
1 on any failure.  Reads the PORT environment variable (default: 8000).
"""
import os
import sys
import urllib.request

port = os.environ.get("PORT", "8000")
url = f"http://localhost:{port}/health"
try:
    urllib.request.urlopen(url, timeout=5)
    sys.exit(0)
except Exception as exc:
    print(f"Health check failed ({url}): {exc}", file=sys.stderr)
    sys.exit(1)
