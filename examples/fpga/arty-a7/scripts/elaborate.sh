#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 scripts/prepare_sources.py "$@"
bash firmware/build.sh
export COURSIER_REPOSITORIES=https://repo.maven.apache.org/maven2
sbt -Dsbt.override.build.repos=true -Dsbt.repository.config=project/repositories \
  -batch 'runMain fpga.arty.EmitPebbleA7 generated'
python3 scripts/prepare_rtl.py
python3 scripts/make_platform.py
python3 scripts/audit_sources.py
