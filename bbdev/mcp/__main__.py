#!/usr/bin/env python3
"""Run: python -u bbdev/mcp/__main__.py (stdio MCP)."""

from __future__ import annotations

import sys
from pathlib import Path

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from server import main

if __name__ == "__main__":
    main()
