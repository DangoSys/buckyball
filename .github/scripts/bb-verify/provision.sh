#!/usr/bin/env bash
# Per-run provision of the persistent root — same sequence as upstream
# check.yaml's chip-check, all incremental inside the persistent root (hot-tree
# timings are noted in bootstrap-persistent-root.sh). This is HOT-TREE PREWARM
# only: the real verification sequence is generated session-side by bbdev_plan
# from the injected judgement block. Prewarm must target the manifest's chip to
# be useful for that chip's build tree; a manifest without a chip, or a
# persistent root without that chip's directory, goes red — falling back to
# toy would only cover up a manifest/persistent-root mismatch.
#
# Compiler-build handling (anti INFRA-loop): if the branch's write set broke
# the compiler, letting compiler --build's non-zero exit go red here creates a
# loop — provision red -> INFRA -> re-dispatch -> same compiler error — while
# the compiler build is CONTENT this gate round must adjudicate. So, only when
# provision_exempt=true (see verdict-inputs.sh: compilerTouched AND this
# round's plan re-runs the compiler build), the failure is demoted: rc
# captured with set +e, the step stays green, and a note hands the call to the
# gate round. Step-level continue-on-error is NOT used: it would also swallow
# real infrastructure failures of nix build / config --install (unrelated to
# the compiler write set, must go red), and with set -e the note line would
# never be reached.
#
# Workload handling (same-flag-skip, exempt-only): workload clean/build
# depends on compiler artifacts. compiler broken => workload necessarily
# broken (same error; rerunning would just pre-stage the gate round's
# adjudication, possibly red), but workload broken =/> compiler broken — so
# the skip applies only when the exemption fired AND the compiler build
# actually failed. In every other case workload runs and its non-zero exit is
# genuinely red.
set -euo pipefail

cd "$BB_VERIFY_ROOT"
export SCRIPTS="$HARNESS_PKG/scripts"

PROVISION_CHIP=$(node --input-type=module -e '
  import { readFileSync, existsSync } from "node:fs"
  const { parseManifest } = await import(`file://${process.env.SCRIPTS}/manifest.mjs`)
  const body = JSON.parse(readFileSync(`${process.env.GITHUB_WORKSPACE}/context.json`, "utf8")).body
  const chip = parseManifest(body).chip
  if (chip === undefined || !existsSync(`${process.env.BB_VERIFY_ROOT}/examples/chips/${chip}`)) {
    console.error(`provision: manifest chip missing or absent from the persistent root ("${chip ?? "<none>"}") -- cannot prewarm, red not green`)
    process.exit(1)
  }
  process.stdout.write(chip)
')
echo "provision chip=${PROVISION_CHIP} (prewarm only; the verification sequence's chip comes from the injected judgement block)"
echo "compilerTouched=${COMPILER_TOUCHED:-<unset>} provisionExempt=${PROVISION_EXEMPT:-<unset>} (verdict-inputs step outputs)"

echo "::group::nix build"; time nix build; echo "::endgroup::"
echo "::group::bbdev config --install"
time nix develop -c bbdev config --install
echo "::endgroup::"

COMPILER_RC=0
if [ "${PROVISION_EXEMPT:-false}" = "true" ]; then
  echo "::group::bbdev compiler --build (incremental; provision_exempt=true -> failure handed to the gate round)"
  set +e
  time nix develop -c bbdev compiler --build "--chip ${PROVISION_CHIP}"
  COMPILER_RC=$?
  set -e
  if [ "$COMPILER_RC" -ne 0 ]; then
    echo "compiler build failure handed to the gate round (provision_exempt=true: this branch touches" \
         "ball/core's compiler/** and this round's plan re-runs config install -> compiler build itself," \
         "so the compiler build is content the gate round adjudicates, not CI infrastructure; exit=${COMPILER_RC})."
  fi
  echo "::endgroup::"
else
  echo "::group::bbdev compiler --build (incremental; provision_exempt=false -> failure is genuinely red)"
  time nix develop -c bbdev compiler --build "--chip ${PROVISION_CHIP}"
  echo "::endgroup::"
fi

echo "::group::bbdev workload --clean + --build (incremental)"
if [ "$COMPILER_RC" -ne 0 ]; then
  echo "workload --clean/--build skipped (same-flag-skip): the compiler build already failed this round and" \
       "was handed to the gate round; workload depends on compiler artifacts and would hit the same error," \
       "pre-staging the gate round's call as a provision red — skipped to keep a single adjudication channel."
else
  time nix develop -c bbdev workload --clean "--chip ${PROVISION_CHIP}"
  time nix develop -c bbdev workload --build "--chip ${PROVISION_CHIP}"
fi
echo "::endgroup::"
