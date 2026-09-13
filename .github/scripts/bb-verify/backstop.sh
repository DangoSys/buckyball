#!/usr/bin/env bash
# Hosted backstop (report job). Covers exactly one scenario: the runner died
# as a whole (OOM/offline), so the verify job's Post terminal commit status
# step never ran and the target sha has no terminal bb-verify status
# (success|failure|error) -> post state=error.
# An existing terminal state (the in-runner step is the only authoritative
# terminal author) or a failed probe (API flake/rate limit) both stand down —
# fail-closed, never overwrites an in-runner verdict. Runs on a GitHub-hosted
# runner: occupies no self-hosted slot, never touches the persistent root,
# only reads/writes the commit status API.
set +e

# Backstop target sha = the verify job's measured HEAD (prepare's resolved_sha
# output); falls back to the sha input when prepare never ran. Both empty =
# even the target is unknown — stand down.
SHA="$VERIFY_RESOLVED_SHA"; [ -n "$SHA" ] || SHA="$SHA_INPUT"
if [ -z "$SHA" ]; then
  echo "report: sha unknown (prepare never resolved, no sha input) -- standing down"
  exit 0
fi

# Existing terminal state -> stand down; probe failure -> stand down
# (fail-closed).
if ! EXISTING=$(gh api "repos/$GITHUB_REPOSITORY/statuses/$SHA" \
  --jq '[.[] | select(.context=="bb-verify")] | .[0].state // empty' 2>/dev/null); then
  echo "report: status probe failed (API failure) -- standing down (fail-closed)"
  exit 0
fi
case "$EXISTING" in
  success|failure|error)
    echo "report: terminal status '$EXISTING' already on ${SHA:0:8} -- standing down"
    exit 0 ;;
esac

# Only pending / no-status remain: verify finished without a verdict =
# infrastructure-side anomaly.
gh api "repos/$GITHUB_REPOSITORY/statuses/$SHA" \
  -f context=bb-verify -f state=error \
  -f target_url="$JOB_URL" \
  -f description="bb-verify: no terminal status (verify job conclusion=$VERIFY_STATUS; runner may have died) -- re-dispatch" \
  && echo "report: posted backstop state=error on ${SHA:0:8}" \
  || echo "report: post failed (best-effort)"
