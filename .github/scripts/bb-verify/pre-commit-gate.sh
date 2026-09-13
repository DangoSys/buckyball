#!/usr/bin/env bash
# Upstream check.yaml pre-commit parity: `nix develop -c pre-commit run
# --all-files` (upstream also passes --show-diff-on-failure --color=always;
# here output lands in a log file and the failure path tails it). Like
# upstream, this enters the dev shell directly without a preceding nix build
# (pre-commit is a devShell tool, seconds-fast on a warm nix store).
#
# bb-verify-task.md is deleted first: it is dispatch metadata, already
# consumed into context.json by the Resolve context step — after deletion,
# pre-commit checks the tree as the author wrote it. The deletion happens
# before the read-only gate's baseline snapshot, so it (and any hook
# auto-fixes) is absorbed into the baseline.
#
# A red gate is a CONTENT failure, not INFRA: runner and tools are fine, the
# branch itself does not pass lint (hook auto-fixer rewrites also exit
# non-zero — that is upstream's failure semantics, deliberately not worked
# around). The always() chain (Finalize/Archive/Post terminal status) then
# settles the run with verdict.txt=FAIL and terminal status=failure.
set -euo pipefail

cd "$BB_VERIFY_ROOT"
rm -f bb-verify-task.md
LOG="$RUNNER_TEMP/pre-commit.log"
set +e
nix develop -c pre-commit run --all-files > "$LOG" 2>&1
RC=$?
set -e

if [ "$RC" -ne 0 ]; then
  printf 'VERDICT: FAIL\n' > "$VERIFY_OUT/verdict.txt"
  {
    echo "# bb-verify: pre-commit gate FAILED"
    echo
    echo "- ref: $REF"
    echo "- head sha: $RESOLVED_SHA"
    echo "- layer: $LAYER"
    echo "- gate: \`nix develop -c pre-commit run --all-files\` (upstream check.yaml pre-commit equivalent), exit=$RC"
    echo
    echo "## hook output tail (full output in this step's CI log)"
    echo '```'
    tail -n 80 "$LOG"
    echo '```'
  } > "$VERIFY_OUT/report.md"
  echo "::error::pre-commit gate FAILED (exit=$RC) -- VERDICT: FAIL (the verified content did not pass the real gate); report.md in the bb-verify-verdict artifact"
  cat "$LOG"
  exit 1
fi
echo "pre-commit gate: PASS"
