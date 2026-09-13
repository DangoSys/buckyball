#!/usr/bin/env bash
# Commit-status handshake for context=bb-verify — the channel callers poll.
#   post-status.sh pending  — posted right after prepare, once the sha is known
#   post-status.sh          — terminal state; the sole source of truth is
#                             $VERIFY_OUT/verdict.txt (PASS -> success,
#                             FAIL -> failure, anything else -> error)
#
# Reporting channel only: this script never fails the job (callers also set
# continue-on-error). If the runner dies before the terminal post, the hosted
# report job (backstop.sh) covers it — it is fail-closed and never overwrites
# a terminal state written here.
set +e

MODE="${1:-terminal}"
if [ -z "${RESOLVED_SHA:-}" ]; then
  echo "post-status: RESOLVED_SHA unknown (prepare never completed) -- standing down (hosted report job covers)"
  exit 0
fi

TARGET_URL="$GITHUB_SERVER_URL/$GITHUB_REPOSITORY/actions/runs/$GITHUB_RUN_ID"
if [ "$MODE" = "pending" ]; then
  STATE=pending
  DESC="bb-verify: run started (ref ${REF}, layer ${LAYER})"
else
  V=$(sed -n -E 's/^VERDICT:[[:space:]]*(PASS|FAIL|INFRA).*/\1/p' "$VERIFY_OUT/verdict.txt" 2>/dev/null | head -1)
  case "$V" in
    PASS) STATE=success ;;
    FAIL) STATE=failure ;;
    *) STATE=error; V="${V:-INFRA}" ;;
  esac
  DESC="bb-verify: ${V} (ref ${REF}, layer ${LAYER})"
fi

gh api "repos/$GITHUB_REPOSITORY/statuses/$RESOLVED_SHA" \
  -f context=bb-verify -f state="$STATE" \
  -f target_url="$TARGET_URL" \
  -f description="$DESC" \
  && echo "post-status: $STATE on ${RESOLVED_SHA:0:8}" \
  || echo "post-status: gh api failed (best-effort; hosted report job covers)"
