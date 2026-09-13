#!/usr/bin/env bash
# Set up the dsh side of the verification session:
#   1) stage the verify-runner plugin from the local checkout and build it
#      (cp -a picks up uncommitted new files — exactly what verification
#      material needs; node_modules/lib are reinstalled/rebuilt so CI verifies
#      a clean build);
#   2) prepare a fresh DSH_HOME with the LLM provider settings (a fresh
#      DSH_HOME without settings.yaml cannot resolve an LLM — Phase-0 lesson;
#      the file carries no key, the key is injected via the DEEPSEEK_API_KEY
#      env var only);
#   3) install the plugin into the headless profile;
#   4) write the plugin config (cordis.patch.yml).
set -euo pipefail

DSH_PLUGIN_DIR="$RUNNER_TEMP/dsh-plugin"
mkdir -p "$DSH_PLUGIN_DIR"
cp -a "$HARNESS_PKG" "$DSH_PLUGIN_DIR/verify-runner"
rm -rf "$DSH_PLUGIN_DIR/verify-runner/node_modules" "$DSH_PLUGIN_DIR/verify-runner/lib"
cd "$DSH_PLUGIN_DIR/verify-runner"
pnpm install
pnpm build
echo "DSH_PLUGIN_DIR=$DSH_PLUGIN_DIR" >> "$GITHUB_ENV"

DSH_LOCAL="$RUNNER_TEMP/dsh-home"
mkdir -p "$DSH_LOCAL"
cp "$DSH_PLUGIN_DIR/verify-runner/ci/settings.ci.yaml" "$DSH_LOCAL/settings.yaml"
# GITHUB_ENV only applies to LATER steps; this script's own config write below
# still needs the variable in-process.
export DSH_HOME="$DSH_LOCAL"
echo "DSH_HOME=$DSH_LOCAL" >> "$GITHUB_ENV"

cd "$DSH_DIR"
pnpm dsh plugin --profile headless add "$DSH_PLUGIN_DIR/verify-runner"

# repoPath points at the persistent root (prepared to ref head + provisioned)
# — verify-runner's bbdev commands resolve it as their cwd (nix develop -c
# bbdev). The tool-workload-integration entry pre-wires the config-only
# credential channel: the plugin reads hfToken from config, not env. That
# plugin is not staged in CI today (the session runs HF_HUB_OFFLINE=1 with a
# preset model cache) and the loader only warns on patch lines for uninstalled
# plugins — the line takes effect the day the plugin enters CI. hfEndpoint
# stays at the default https://huggingface.co.
mkdir -p "$DSH_HOME/profiles/headless"
cat > "$DSH_HOME/profiles/headless/cordis.patch.yml" <<EOF
- id: tool-verify-runner
  config:
    repoPath: $BB_VERIFY_ROOT
    taskDir:  $RUNNER_TEMP/verify-tasks
- id: tool-workload-integration
  config:
    repoPath: $BB_VERIFY_ROOT
    hfToken: "$HF_TOKEN"
EOF
