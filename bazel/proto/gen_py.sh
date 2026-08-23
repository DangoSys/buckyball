#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/../../" && pwd)"
proto="$root/bazel/proto"
out="$root/bazel/configparse"
protoc -I"$proto" --python_out="$out" "$proto/chip_bundle.proto"
echo "wrote $out/chip_bundle_pb2.py"
