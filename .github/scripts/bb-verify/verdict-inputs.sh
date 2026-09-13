#!/usr/bin/env bash
# Deterministic verdict inputs: the five verdict pre-judgements (stage
# inference, evidence-manifest schema validation, model-binding three-source
# check, slices evidence check, perf previous-round probe) are produced by the
# pure-node scripts in the verify-runner package and injected into
# task-prompt.md — the session's prompt.ts consumes them, it never re-derives
# them. A non-zero exit here is a judgement-pipeline failure: red, never a
# fake green (a PRE-FAIL inside the block lets the session answer FAIL
# directly, saving a wasted full batch).
#
# Runs the scripts straight from the local checkout ($HARNESS_PKG/scripts):
# pure node, no build step; the cp -a staging happens later in setup-dsh.sh.
# Must run AFTER prepare: the manifest/binding tree checks inspect the ref
# head, and context.json (Resolve context step) needs the fetched objects.
set -euo pipefail

export SCRIPTS="$HARNESS_PKG/scripts"
cd "$GITHUB_WORKSPACE"

# infer-stage lands on disk first: its output goes into the task-prompt.md
# judgement block ([stage] section) AND feeds the two step outputs below that
# drive provision.sh's compiler-build handling.
STAGE_BLOCK="$RUNNER_TEMP/infer-stage.out"
node "$SCRIPTS/infer-stage.mjs" --context context.json > "$STAGE_BLOCK"

# compilerTouched=true means the branch's cumulative write set touches
# ball/core's compiler/** — the compiler build itself is then content this
# gate round must adjudicate, not CI infrastructure.
COMPILER_TOUCHED=$(sed -n -E 's/^[[:space:]]*compilerTouched:[[:space:]]*(true|false).*/\1/p' "$STAGE_BLOCK")
case "$COMPILER_TOUCHED" in
  true|false) : ;;
  *)
    # Unparseable = judgement block format broke (infer-stage crashed or its
    # output surface changed). Fail closed: silently treating it as false
    # would re-create the "compiler write set broke the build -> provision
    # red -> INFRA -> re-dispatch" loop this output exists to eliminate.
    echo "::error::could not extract compilerTouched from infer-stage output (got '${COMPILER_TOUCHED:-<empty>}') -- judgement pipeline failure, red not green"
    exit 1 ;;
esac
echo "compiler_touched=$COMPILER_TOUCHED" >> "$GITHUB_OUTPUT"

# provision_exempt: demoting a compiler-build failure in provision.sh is only
# sound when THIS round's gate plan re-runs the compiler build itself — the
# exemption's extra condition over compilerTouched.
# This case table MIRRORS the compilerPrerequisite() call sites in
# src/tools/bbdev-plan.ts — that file is the single source of truth (CI cannot
# read plugin code at runtime), so any change there must be mirrored here.
# Combos with a config-install -> compiler-build prerequisite: no-phase
# baselines (workload/ball/chip), ball/c-bemu, ball/rtl, chip/skeleton,
# chip/slices, chip/integrate. The one legal combo WITHOUT the prerequisite is
# chip/bind, and bind is deliberately NOT exempt: its `workload --build
# --model` still consumes compiler artifacts (bb-tests/workloads/CMakeLists.txt
# points BUDDY_BINARY_DIR at $BUDDY_MLIR_BUILD_DIR/bin for buddy-opt /
# buddy-translate / buddy-llc), so an exempted bind round could go green on a
# stale warm-tree compiler with the branch's compiler breakage unadjudicated.
# Stage comes from [stage] effective, phase from the manifest declaration —
# if either is undeterminable the exemption does NOT apply (it needs positive
# evidence that this round re-runs the compiler build).
EFFECTIVE=$(sed -n -E 's/^[[:space:]]*effective:[[:space:]]*(workload|ball|chip)[[:space:]]*$/\1/p' "$STAGE_BLOCK")
MANIFEST_PHASE=$(node --input-type=module -e '
  import { readFileSync } from "node:fs"
  const { parseManifest } = await import(`file://${process.env.SCRIPTS}/manifest.mjs`)
  const body = JSON.parse(readFileSync(`${process.env.GITHUB_WORKSPACE}/context.json`, "utf8")).body
  process.stdout.write(parseManifest(body).phase ?? "")
')
PROVISION_EXEMPT=false
if [ "$COMPILER_TOUCHED" = "true" ]; then
  case "${EFFECTIVE:-none}/${MANIFEST_PHASE:-none}" in
    workload/none|ball/none|chip/none|ball/c-bemu|ball/rtl|chip/skeleton|chip/slices|chip/integrate)
      PROVISION_EXEMPT=true ;;
  esac
fi
echo "provision_exempt=$PROVISION_EXEMPT" >> "$GITHUB_OUTPUT"
echo "verdict_inputs: compilerTouched=$COMPILER_TOUCHED effective=${EFFECTIVE:-<undetermined>} phase=${MANIFEST_PHASE:-<none>} -> provision_exempt=$PROVISION_EXEMPT (step outputs; provision.sh's compiler-build handling)"

# Runner EDA capability probe — the precondition fact for the complete layer's
# regression.yml EDA lanes. dc_shell/vcs/vivado are host installs outside the
# nix env (upstream regression.yml sources the host shell rc); probe this
# job's actual PATH, same measure as the plan tool (bbdev-plan.ts reuses
# bbdev-common's findOnPath, also PATH-scanning). A lane that should run while
# this says no -> the plan records it in blockedLanes and concludes INFRA,
# naming the missing tool.
probe_eda() { command -v "$1" >/dev/null 2>&1 && echo yes || echo no; }
EDA_CAP="dc_shell=$(probe_eda dc_shell) vcs=$(probe_eda vcs) vivado=$(probe_eda vivado)"
echo "verdict_inputs: runner EDA capability: $EDA_CAP"

{
  # Agent-facing session contract (the session's prompt.ts consumes this
  # verbatim — keep the wording stable): there is no PR; the verdict lands on
  # the stdout VERDICT line and in $VERIFY_OUT/report.md; gh pr comment /
  # gh pr review are forbidden (nothing to post to, a hollow run just wastes
  # turns).
  echo "验证临时分支 $REF（head ${RESOLVED_SHA:0:8}）。验证层级：$LAYER（merge=门禁层 ≈ check.yaml；complete=重验证 ≈ check.yaml + regression.yml EDA 车道）。"
  echo "runner EDA capability: $EDA_CAP（complete 层 EDA 车道的工具前置；dc/uvm/bebop-p2e 车道该跑而此处为 no → plan 会把该 lane 记进 blockedLanes，结论 INFRA 点名缺失工具）"
  echo
  echo "重要：本次验证没有 PR。禁止运行 gh pr comment / gh pr review。"
  echo "最终结论必须同时落在两处："
  echo "  1) 在 stdout 单独打印一行 VERDICT: PASS 或 VERDICT: FAIL（CI 门禁只认这一行）；"
  echo "  2) 把完整 markdown 报告写入 $VERIFY_OUT/report.md。"
  echo "分支上下文（title=commit subject，body=仓库根 bb-verify-task.md 或 commit body，files=与 origin/main 的 diff）如下："
  cat context.json
  echo
  echo "===== CI 确定性判定（脚本注入，直接消费，不得重新判定）====="
  cat "$STAGE_BLOCK"
  node "$SCRIPTS/validate-manifest.mjs" --context context.json --repo-root "$BB_VERIFY_ROOT"
  node "$SCRIPTS/binding-check.mjs" --context context.json --repo-root "$BB_VERIFY_ROOT"
  node "$SCRIPTS/slices-verify.mjs" --context context.json
  node "$SCRIPTS/probe-loop-check.mjs" --context context.json
} > task-prompt.md
