#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/../../" && pwd)"
proto="$root/bazel/proto"
py_out="$root/bazel/configparse"
java_out="$proto/generated/java"
mkdir -p "$java_out"
protoc -I"$proto" --python_out="$py_out" "$proto/chip_bundle.proto"
protoc -I"$proto" --java_out="$java_out" "$proto/chip_bundle.proto"
echo "wrote $py_out/chip_bundle_pb2.py"
echo "wrote $java_out/buckyball/config/*.java"
