#!/usr/bin/env bash

set -euo pipefail

BBDIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)

usage() {
  echo "Usage: ${0} [OPTIONS] "
  echo ""
  echo "Helper script to fully initialize repository that wraps other scripts."
  echo "By default it initializes/installs things in the following order:"
  echo "   0. Nix environment (then git submodules, using nix git)"
  echo "   1. bbdev install"
  echo "   2. Compiler installation"
  echo "   3. RTL pre-compile sources"
  echo "   4. bb-tests pre-compile sources"
  echo "   5. waveform-mcp build"
  echo "   6. bebop build"
  echo "   7. verify build"
  echo "   8. pre-commit hooks installation"
  echo "   9. register project MCP"
  echo ""
  echo "Selected steps require earlier steps to have completed already."
  echo ""
  echo "Options"
  echo "  -h     : Display this message"
  echo "  -o N   : Run only step N. Repeat to select multiple steps; step 0 is not automatic."
  echo "  -s N   : Skip step N. Repeat to skip multiple steps; cannot combine with -o."
}

ONLY_LIST=()
SKIP_LIST=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    -h)
      usage
      exit 0 ;;
    -o | -s)
      option=$1
      shift
      if [ "$#" -eq 0 ] || [[ ! "$1" =~ ^[0-9]$ ]]; then
        echo "Error: ${option} requires a step number from 0 to 9" >&2
        exit 2
      fi
      case "$option" in
        -o) ONLY_LIST+=("$1") ;;
        -s) SKIP_LIST+=("$1") ;;
      esac ;;
    * )
      echo "Error: invalid option $1" >&2
      exit 2 ;;
  esac
  shift
done

if [ "${#ONLY_LIST[@]}" -gt 0 ] && [ "${#SKIP_LIST[@]}" -gt 0 ]; then
  echo "Error: -o and -s cannot be combined" >&2
  exit 2
fi

run_step() {
  local value=$1
  local step
  if [ "${#ONLY_LIST[@]}" -gt 0 ]; then
    for step in "${ONLY_LIST[@]}"; do
      [ "$step" = "$value" ] && return 0
    done
    return 1
  fi
  for step in "${SKIP_LIST[@]}"; do
    [ "$step" = "$value" ] && return 1
  done
  return 0
}

function begin_step
{
  thisStepNum=$1;
  thisStepDesc=$2;

  # Color codes
  local BLUE='\033[0;34m'
  local GREEN='\033[0;32m'
  local YELLOW='\033[1;33m'
  local NC='\033[0m' # No Color

  echo -e "${BLUE} ========================================================================="
  echo -e "${GREEN} ==== BUCKYBALL SETUP STEP ${YELLOW}$thisStepNum${GREEN}: ${YELLOW}$thisStepDesc${GREEN} "
  echo -e "${BLUE} ========================================================================="
  echo -e "${NC}"
}

cd "$BBDIR"
if run_step 0 && [ "${BUCKYBALL_SETUP_NIX_BUILT:-0}" != "1" ]; then
  begin_step "0" "Nix environment setup"
  nix build
fi

if [ -z "${IN_NIX_SHELL:-}" ]; then
  REEXEC_ARGS=()
  for only in "${ONLY_LIST[@]}"; do
    REEXEC_ARGS+=(-o "$only")
  done
  for skip in "${SKIP_LIST[@]}"; do
    REEXEC_ARGS+=(-s "$skip")
  done
  export BUCKYBALL_SETUP_NIX_BUILT=1
  exec nix develop --command bash "$BBDIR/scripts/nix/build-all.sh" "${REEXEC_ARGS[@]}"
fi

if run_step 0; then
  begin_step "0" "Git submodules"
  "$BBDIR/scripts/nix/download.sh"
fi

if run_step "1"; then
  begin_step "1" "bbdev install"

  echo "Installing bbdev Python dependencies..."
  cd "$BBDIR/bbdev/api"
  uv venv .venv --python python3 --seed
  uv pip install --python .venv/bin/python -r pyproject.toml

  cd "$BBDIR"
  bbdev config --install
fi

if run_step "2"; then
  begin_step "2" "Compiler installation"
  cd "$BBDIR"
  bbdev compiler --build '--chip toy'
fi

if run_step "3"; then
  begin_step "3" "RTL source pre-compile"
  bbdev verilator --verilog '--chip toy'
fi

if run_step "4"; then
  begin_step "4" "bb-tests pre-compile sources"
  bbdev workload --build '--chip toy'
fi

if run_step "5"; then
  begin_step "5" "waveform-mcp build"
  cd "$BBDIR/thirdparty/waveform-mcp"
  cargo build --release
fi

if run_step "6"; then
  begin_step "6" "bebop build"
  cd "$BBDIR/bebop"
  nix build
  nix develop -c echo "bebop built successfully"


fi

if run_step "7"; then
  begin_step "7" "verify build"
  cd "$BBDIR/verify"
  nix develop -c echo "verify built successfully"
fi

if run_step "8"; then
  begin_step "8" "pre-commit hooks installation"
  cd "$BBDIR"
  pre-commit install --overwrite --hook-type pre-commit
  # Replace with wrapper so git commit gets nix env (result/bin in PATH)
  cp "${BBDIR}/scripts/pre-commit-hook.sh" "${BBDIR}/.git/hooks/pre-commit"
fi

if run_step "9"; then
  begin_step "9" "register project MCP and Skills"
  bash "${BBDIR}/.agents/mcps/scripts/install.sh"
  bash "${BBDIR}/.agents/skills/scripts/install.sh"
fi

begin_step "END" "Setup completed successfully!"
