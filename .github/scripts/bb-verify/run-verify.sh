#!/usr/bin/env bash
# Run the headless verify-runner session, then adjudicate this step's exit
# code from the captured stdout (verdict-gate) and write verdict.txt — the
# Post terminal commit status step's sole source of truth.
#
# Why the indirection: dsh headless's process exit code only reflects whether
# the session completed normally (completed -> 0), not PASS/FAIL (src/prompt.ts
# contract) — the verdict is expressed by the VERDICT line of the final
# answer. So the session runs to completion first, then:
#   first ^VERDICT: (PASS|FAIL) line + session exit 0 -> verdict.txt=PASS, exit 0
#   VERDICT: FAIL                              -> verdict.txt=FAIL, exit 1
#   unparseable                                -> verdict.txt=INFRA, exit 1
# stdout is the only source of the verdict; there is no other surface to fall
# back to. PASS with a non-zero session exit is untrustworthy: no verdict.txt
# write, exit 1, and Finalize verdict files lands INFRA (terminal status=error).
set -euo pipefail

VERIFY_STDOUT="$RUNNER_TEMP/verify-stdout.log"
VERIFY_STDERR="$RUNNER_TEMP/verify-stderr.log"

cd "$DSH_DIR"
# HF network paths are guaranteed-dead in CI: the harness http-proxy policy
# rewrites no_proxy for child processes with a bracketed `[::1]` that httpx
# 0.28.1 fails to parse, and a step-level export does not help (the harness
# rewrites it). CI model caches are preset (~/.cache/huggingface), so go fully
# offline: huggingface_hub's offline mode short-circuits before any HTTP
# client is constructed. Consequence: a non-preset model fails deterministically
# with OfflineModeIsEnabled instead of wandering onto the network.
export HF_HUB_OFFLINE=1

# EDA session sandbox: empty EDA_SESSION_BINDS = run unsandboxed (the upstream
# runner has /home/bb-runner/Code/eda natively). The fork deployment
# materializes that tree + the lc_shell wrapper + lc's libstdc++ swap + a
# files-only nsswitch.conf (libnss_systemd SEGVs the R-2020.09 lc_shell under
# bwrap; files-only NSS removes that module from the lookup chain).
SANDBOX=()
if [ -n "${EDA_SESSION_BINDS:-}" ]; then
  SANDBOX=(bwrap --dev-bind / / --tmpfs /home --bind "$HOME" "$HOME")
  IFS=';' read -ra BIND_PAIRS <<< "$EDA_SESSION_BINDS"
  for pair in "${BIND_PAIRS[@]}"; do
    [ -n "$pair" ] || continue
    src=${pair%%>*}
    dst=${pair#*>}
    SANDBOX+=(--dir "$(dirname "$dst")")
    if [ -f "$src" ]; then
      SANDBOX+=(--ro-bind "$src" "$dst")
    else
      SANDBOX+=(--bind "$src" "$dst")
    fi
  done
  SANDBOX+=(--)
fi

set +e
"${SANDBOX[@]}" pnpm dsh --profile headless "$(cat "$GITHUB_WORKSPACE/task-prompt.md")" \
  > "$VERIFY_STDOUT" 2> "$VERIFY_STDERR"
DSH_RC=$?
set -e

LINE=$(grep -m1 -E '^VERDICT:[[:space:]]*(PASS|FAIL)' "$VERIFY_STDOUT" || true)
VERDICT=$(printf '%s\n' "$LINE" | sed -n -E 's/^VERDICT:[[:space:]]*(PASS|FAIL).*/\1/p')

# Fold the session stdout/stderr into collapsible ::group::s and write a lane
# summary ($GITHUB_STEP_SUMMARY). The verdict-gate reads the $VERIFY_STDOUT
# FILE, unaffected by this presentation wrapping.
# mask_wf: session output lands verbatim in the Actions log, where a leading
# "::name::" would be parsed as a workflow command — session reports quote the
# verified party's prose/tool output, which would otherwise be a channel for
# forging ::error::/::group:: or even ::stop-commands::. Rewrites the command
# shape at line start to "[name] ": semantics preserved, annotation surface
# neutralized. Presentation only; the gate still reads the raw file.
mask_wf() { sed -E 's/^::([a-zA-Z][a-zA-Z0-9-]*)([[:space:]][^:]*)?::/[\1] /'; }
echo "::group::verify-runner session stdout"
mask_wf < "$VERIFY_STDOUT"
echo "::endgroup::"
echo "::group::verify-runner session stderr"
mask_wf < "$VERIFY_STDERR" >&2
echo "::endgroup::"

# The step summary is markdown: backticks in LINE would break inline code, `|`
# the table, and ``` in judgement-block fragments would close the code fence
# early — strip before rendering.
LINE_SAFE=$(printf '%s' "${LINE:-<none>}" | tr -d '`\r' | tr '|' '/' | cut -c1-120)
{
  echo "### bb-verify — ref $REF · ${RESOLVED_SHA:0:8} · layer=$LAYER"
  echo ""
  echo "| item | value |"
  echo "| --- | --- |"
  echo "| dsh session exit | \`$DSH_RC\` |"
  echo "| VERDICT (stdout) | \`$LINE_SAFE\` |"
  echo ""
  echo "#### injected judgement block (lane basis)"
  echo '```'
  grep -aE '^\[|effective|chip|phase|layer|PRE-FAIL|VERDICT' \
    "$GITHUB_WORKSPACE/task-prompt.md" 2>/dev/null | head -60 \
    | tr -d '\r' | sed -E 's/^ *```.*$//'
  echo '```'
} >> "$GITHUB_STEP_SUMMARY" || true

# ==== verdict-gate (the only adjudication point, fail-fast) ====
echo "verdict-gate: parsed VERDICT=${VERDICT:-<none>}, dsh session exit=${DSH_RC}"
case "$VERDICT" in
  PASS)
    if [ "$DSH_RC" -ne 0 ]; then
      echo "::error::VERDICT says PASS but dsh session exited ${DSH_RC} -- failing (no fake green)"
      exit 1
    fi
    printf 'VERDICT: PASS\n' > "$VERIFY_OUT/verdict.txt"
    echo "verdict-gate: PASS -- job green"
    ;;
  FAIL)
    printf 'VERDICT: FAIL\n' > "$VERIFY_OUT/verdict.txt"
    echo "::error::VERDICT: FAIL -- verification failed (see report.md in the bb-verify-verdict artifact)"
    exit 1
    ;;
  *)
    printf 'VERDICT: INFRA\n' > "$VERIFY_OUT/verdict.txt"
    echo "::error::no 'VERDICT: PASS|FAIL' line in headless output -- treated as INFRA (no fake green)"
    exit 1
    ;;
esac
