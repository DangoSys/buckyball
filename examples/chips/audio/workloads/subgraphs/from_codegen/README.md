# Regenerated Whisper subgraphs (framework path)

Produced by:

```bash
nix develop -c bash -lc '
  export PYTHONPATH=$PWD/compiler/thirdparty/buddy-mlir/build/python_packages:$PYTHONPATH
  export WHISPER_MODEL_PATH=<local-whisper-base-snapshot>
  python3 bb-tests/workloads/src/ModelTest/e2e/models/models/Whisper/codegen/import-whisper.py \
    --spec bb-tests/workloads/src/ModelTest/e2e/models/models/Whisper/specs/base.json \
    --output-dir examples/chips/audio/workloads/subgraphs/from_codegen \
    --layer-partitioned \
    --no-partition-decoder
'
```

Contents:

| Path | Meaning |
|------|---------|
| `sg_dap.mlir` | DAP preprocess (from Buddy `DAP-extend.mlir`) |
| `sg_encoder/` | Whole encoder stage |
| `sg_decoder/` | Whole decoder stage |
| `layer_partitioned/encoder/subgraph0_encoder{0..6}.mlir` | Encoder attention-boundary partitions |
| `layer_partitioned/partition_manifest.json` | Counts + `audio_slice_groups` |

`*.data` weight blobs are **not** kept in-tree (regenerate with the command above).

## Decoder note

`--partition-decoder` is off by default: `PartitionedGraphDriver` child decoder partitions currently fail lowering (`attn_mask` symbol None in tosa). Until that mainline gap is fixed, `dec_0`/`dec_1` share `sg_decoder/` stage IR; encoder homogeneous slices use the partition groups in the manifest.
