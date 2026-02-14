#!/usr/bin/env bash
# Wrapper to run pre-commit with Nix env (result/bin in PATH).
# Git hooks run without nix develop, so we must load the env manually.

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${HERE}/../.."
export PATH="${REPO_ROOT}/result/bin:${PATH}"

exec pre-commit hook-impl --config="${REPO_ROOT}/.pre-commit-config.yaml" \
  --hook-type=pre-commit --hook-dir "$HERE" -- "$@"
