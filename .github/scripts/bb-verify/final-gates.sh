#!/usr/bin/env bash
# Green-path final gates.
#
# 1) Verdict artifacts complete: verdict.txt must parse to PASS|FAIL|INFRA
#    (on the green path it is necessarily PASS — the FAIL/INFRA paths are
#    already red and never reach here; this gate guards the "file lost /
#    written bad" fake-green channel), and report.md must exist and be
#    non-empty. No placeholders: a session that did not write its report did
#    not fulfil its contract — judge INFRA (rewriting verdict.txt makes the
#    terminal status error) and go red.
# 2) Read-only gate: the verification session is read-only on the persistent
#    root — the tracked diff before/after the session is compared byte-for-byte
#    (baseline taken by the Snapshot step; pre-existing submodule build residue
#    is absorbed by the baseline, new session writes always show up).
set -euo pipefail

V=$(sed -n -E 's/^VERDICT:[[:space:]]*(PASS|FAIL|INFRA)[[:space:]]*$/\1/p' "$VERIFY_OUT/verdict.txt" 2>/dev/null | head -1)
if [ -z "$V" ]; then
  printf 'VERDICT: INFRA\n' > "$VERIFY_OUT/verdict.txt"
  echo "::error::verdict-gate: $VERIFY_OUT/verdict.txt missing or unparseable -- INFRA (no fake green)"
  exit 1
fi
if [ ! -s "$VERIFY_OUT/report.md" ]; then
  printf 'VERDICT: INFRA\n' > "$VERIFY_OUT/verdict.txt"
  echo "::error::verdict-gate: $VERIFY_OUT/report.md missing or empty -- session failed its report contract -- INFRA (no fake green)"
  exit 1
fi
echo "verdict-gate: verdict.txt=$V, report.md present ($(wc -c < "$VERIFY_OUT/report.md") bytes)"

git -C "$BB_VERIFY_ROOT" diff HEAD > "$RUNNER_TEMP/post-verify.diff"
if ! cmp -s "$RUNNER_TEMP/pre-verify.diff" "$RUNNER_TEMP/post-verify.diff"; then
  echo "::error::persistent root tracked diff changed during the session (verifier must be read-only)"
  diff "$RUNNER_TEMP/pre-verify.diff" "$RUNNER_TEMP/post-verify.diff" | head -50 || true
  exit 1
fi
echo "readonly-gate: persistent root diff unchanged"
