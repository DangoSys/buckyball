# bb-verify

Dispatch-only CI that verifies arbitrary temp branches (`verify/*`) pushed to
this repo — no PRs, no `pull_request` trigger. One workflow serves all callers:
the BB Harness verify-runner plugin (or a human) pushes a temp branch,
dispatches this workflow, and collects the verdict from the commit status and
the verdict artifact.

Step logic lives in `.github/scripts/bb-verify/` — one script per step, named
after it.

## Trigger

`workflow_dispatch` on the default branch, with three inputs:

- `ref` — the temp branch. Must match `^verify/[A-Za-z0-9._-]+$`; anything
  else (e.g. `main`) is rejected.
- `sha` — expected branch head (40-hex). Compared with the actual HEAD under
  the repo lock; a branch that moved between push and dispatch goes red
  instead of verifying the wrong code.
- `layer` — `merge`: fast gate (pre-commit + bemu batches, roughly the
  `check.yaml` surface). `complete`: heavy re-verify, adding pk-tests /
  verilator and — when the host has `dc_shell`/`vcs`/`vivado` — the
  `regression.yml` EDA lanes. A lane that should run but lacks its tool is
  reported as INFRA (naming the tool), never silently skipped.

Optionally, the branch can carry a `bb-verify-task.md` at the repo root
describing the task; it becomes the session's brief (and is deleted before the
pre-commit gate so hooks see the tree as the author wrote it).

## What a run does

Runs on the self-hosted runner against a persistent, warmed checkout
(`BB_VERIFY_ROOT`), serialized by the workflow concurrency group plus an
`flock` around checkout switching. Sequence: validate inputs → prepare the
persistent root (fetch + detached checkout + incremental submodules) →
pre-commit gate (parity with `check.yaml`) → deterministic verdict inputs →
incremental provision (`nix build`, `bbdev config/compiler/workload`) →
headless verification session driving whitelisted `bbdev` commands → gates →
verdict. The session is read-only on the persistent root (tracked diff
compared byte-for-byte before/after).

## Results

- Commit status `bb-verify` on the verified sha: `pending` at start, then
  `success` (PASS) / `failure` (FAIL) / `error` (INFRA or missing verdict). A
  hosted backstop job posts `error` if the runner died before posting a
  terminal status; it never overwrites an existing one.
- Artifact `bb-verify-verdict`: `verdict.txt` (single line, the
  machine-readable source of truth) + `report.md` (full markdown report;
  absent on INFRA, by design).
- INFRA means the CI infrastructure failed, not the verified content:
  re-dispatch.

## Deployment

All machine-specific values live in the workflow's top-level `env` block:
adjust those 10 variables, the `runs-on` label and the PATH-fix lines to your
host — nothing else.

- `BB_VERIFY_ROOT` — persistent checkout, built once by
  `packages/verify-runner/ci/bootstrap-persistent-root.sh` (harness repo) or
  equivalent: clone + submodules + `nix build` + bbdev venv + bebop spike +
  first compiler/workload build.
- `DSH_DIR` / `HARNESS_PKG` — local checkouts of the dsh CLI and the harness
  verify-runner plugin; the verifier logic (`scripts/*.mjs`) loads from there
  and stays versioned with the plugin.
- `VERIFY_OUT` — verdict directory, writable from the dsh session sandbox.
- `RELAY_ADDR` — optional `host:port` of a local LLM relay; empty = direct.
- `SNPSLMD_LICENSE_FILE` / `LM_LICENSE_FILE` — Synopsys license server for
  dc_shell/vcs/lc_shell (regression.yml pattern).
- `EDA_BIN` — dir with dc_shell/vcs wrappers (legacy-lib shims), prepended to
  PATH; empty = tools natively on PATH.
- `EDA_SESSION_BINDS` — `;`-separated `src>dst` bind pairs for the session
  sandbox (bwrap) materializing `/home/bb-runner/Code/eda`; empty = the host
  has that tree natively, no sandbox.
- Secrets: `DEEPSEEK_API_KEY` (required), `HF_TOKEN` (optional, model cache).

Runner host needs nix, gh, node, pnpm on PATH, the bootstrapped persistent
root, and a warm Hugging Face cache (the session runs with
`HF_HUB_OFFLINE=1`). Use a dedicated runner label: the persistent root must
not be shared with `check.yaml`/`regression.yml`, whose `ci_repo_lock.sh`
resets trees.
