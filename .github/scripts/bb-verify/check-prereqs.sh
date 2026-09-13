#!/usr/bin/env bash
# Fail-fast prerequisite probe, before any heavy work: toolchain on PATH,
# local checkouts and one-time bootstrap artifacts present, secret configured,
# optional LLM relay reachable.
set -euo pipefail

for tool in nix gh pnpm node; do
  command -v "$tool" >/dev/null || { echo "::error::$tool not on PATH"; exit 1; }
done
# Record which node/pnpm the job actually resolves — PATH-order bugs here have
# burned us (bare `pnpm dsh` breaks on the flake node; the runner PATH must
# hit ~/.local/bin/node instead).
echo "node at $(command -v node): $(node --version)"
echo "pnpm at $(command -v pnpm): $(pnpm --version)"

for path in \
  "$DSH_DIR" \
  "$HARNESS_PKG/ci/settings.ci.yaml" \
  "$BB_VERIFY_ROOT/.git" \
  "$BB_VERIFY_ROOT/bbdev/api/.venv/bin/motia" \
  "$BB_VERIFY_ROOT/bebop/src/nodes/bemu/native/spike/configure.ac" ; do
  [ -e "$path" ] || {
    echo "::error::missing local path: $path"
    echo "::error::persistent root not bootstrapped or one-time artifacts missing -- run bootstrap-persistent-root.sh once on this host"
    exit 1; }
done

[ -n "$DEEPSEEK_API_KEY" ] || {
  echo "::error::repo secret DEEPSEEK_API_KEY is not configured"; exit 1; }

# Optional local LLM relay (e.g. a UA-rewriting whitelist gate). If configured
# it must be up before the session starts — probe the TCP port and fail
# explicitly here instead of letting the dsh session blow up mid-run.
if [ -n "${RELAY_ADDR:-}" ]; then
  timeout 3 bash -c "</dev/tcp/${RELAY_ADDR%:*}/${RELAY_ADDR#*:}" || {
    echo "::error::LLM relay ($RELAY_ADDR) is not reachable"; exit 1; }
fi
