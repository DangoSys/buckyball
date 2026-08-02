#!/usr/bin/env bash
# Launch buckyball-dev MCP with a clean stdout (required by MCP stdio).
# Always runs from the repository root so nix shellHook / sourceme.sh resolve correctly.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

NIX_BIN="${NIX_BIN:-$(command -v nix || true)}"
if [[ -z "$NIX_BIN" && -x /nix/var/nix/profiles/default/bin/nix ]]; then
  NIX_BIN=/nix/var/nix/profiles/default/bin/nix
fi
if [[ -z "$NIX_BIN" ]]; then
  echo "ERROR: nix not found in PATH" >&2
  exit 1
fi

export NIX_QUIET=1
exec "$NIX_BIN" develop "$ROOT" -c python3 -u "$ROOT/scripts/claude/mcp_server.py"
