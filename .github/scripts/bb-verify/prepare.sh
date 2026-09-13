#!/usr/bin/env bash
# Prepare the persistent root: flock + fetch the target head + detached
# checkout + incremental submodule alignment + sha race guard.
#
# Why not reuse upstream's .github/scripts/ci_repo_lock.sh prepare:
#   1) its repo_reset (git clean -ffd + reset --hard in bbdev/bebop/buddy-mlir)
#      wipes every warmed incremental build artifact (buddy-mlir build/,
#      bb-tests/workloads/build/, .venv, spike, ...), defeating the persistent
#      warmup — test-product cleanliness here is instead guaranteed by the
#      batch commands' --clean-before;
#   2) its lock dir and proxy assumptions are specific to upstream's runner.
#
# Fetch order matters: ref-context.mjs computes files as the name-only diff
# against origin/main, so main must be fresh; and checkout consumes FETCH_HEAD,
# which a later fetch overwrites — fetch main FIRST, keep the target ref in
# the LAST fetch.
set -euo pipefail

mkdir -p "$(dirname "$BB_VERIFY_LOCK")"
exec 9>"$BB_VERIFY_LOCK"
flock 9

git -C "$BB_VERIFY_ROOT" fetch --force origin main
git -C "$BB_VERIFY_ROOT" fetch --force origin "refs/heads/${REF}"
git -C "$BB_VERIFY_ROOT" checkout --detach --force FETCH_HEAD

# Realign first-level submodule pointers to the ref head (seconds-long no-ops
# on the bootstrapped tree). `submodule sync` is required after .gitmodules
# URL changes: update --init does not re-read new URLs for already-initialized
# submodules. Deliberately NOT --recursive: buddy-mlir's nested
# riscv-gnu-toolchain/tt-mlir are gigabytes the verification path never
# touches (the compiler build only needs llvm) — a full recursive clone every
# run would be slow and a single point of network failure.
git -C "$BB_VERIFY_ROOT" submodule sync
git -C "$BB_VERIFY_ROOT" submodule update --init
# buddy-mlir's nested llvm submodule is a hard compiler-build dependency;
# installed at bootstrap time (shallow fetch by sha), topped up only if missing.
if [ ! -f "$BB_VERIFY_ROOT/compiler/thirdparty/buddy-mlir/llvm/llvm/CMakeLists.txt" ]; then
  git -C "$BB_VERIFY_ROOT/compiler/thirdparty/buddy-mlir" submodule update --init llvm
fi

HAVE_SHA=$(git -C "$BB_VERIFY_ROOT" rev-parse HEAD)
# Race guard, inside the flock critical section: a branch re-pushed between
# push and dispatch goes red here instead of verifying the wrong code.
if [ "$HAVE_SHA" != "$EXPECTED_SHA" ]; then
  echo "::error::persistent root at ${HAVE_SHA:0:8}, expected ref head ${EXPECTED_SHA:0:8} (sha input does not match the branch head)"
  exit 1
fi
flock -u 9

# resolved_sha is what the commit-status handshake and the hosted report job
# backstop operate on (consumed via the job's outputs).
echo "resolved_sha=$HAVE_SHA" >> "$GITHUB_OUTPUT"
echo "RESOLVED_SHA=$HAVE_SHA" >> "$GITHUB_ENV"
echo "persistent root at $(git -C "$BB_VERIFY_ROOT" log --oneline -1)"
