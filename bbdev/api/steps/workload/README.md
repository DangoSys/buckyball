# Workload Workflow

Workload build workflow in Buckyball framework, used to build test workloads and benchmark programs.

## API Usage

### `clean`
**Endpoint**: `POST /workload/clean`

**Function**: Clean workload output directory for one chip.

**Parameters**:
- **`chip`** - Required chip name.

**Examples**:
```bash
bbdev workload --clean "--chip toy"
```

### `build`
**Endpoint**: `POST /workload/build`

**Function**: Build workload

**Parameters**:
- **`chip`** - Required chip name. Selects chip-specific workloads.
- **`model`** - Optional model name to build. If omitted, build all workloads.
- **`stable`** - Optional boolean flag. If set, build with stable LLVM Buckyball extensions.
- **`rushB`** - Optional host-native backend: `bemu` or `verilator`. Requires
  `model`. It builds host-native CPU and rushB-lowered accelerator objects.
- **`ctest`** - Build CTest workloads only. Cannot be combined with `model` or `rushB`.
- **`mlirtest`** - Build MLIRTest workloads only. Cannot be combined with `model` or `rushB`.

For chip workloads under paths like `*/chips/<chip>`, only the directory selected by `chip` is synced to `bb-tests/output/<chip>/workloads`.

**Examples**:
```bash
# Build one model with the default xbuckyball path
bbdev workload --build "--chip toy --model lenet"

# Build one model with stable LLVM Buckyball extensions
bbdev workload --build "--chip toy --model lenet --stable"

# Build a model's rushB BEMU runner
bbdev workload --build "--chip pebble --model lenet --rushB bemu"

# Build a model's rushB Verilator runner
bbdev workload --build "--chip pebble --model lenet --rushB verilator"

# Build all workloads
bbdev workload --build "--chip toy"

# Build only CTest workloads
bbdev workload --build "--chip pebble --ctest"

# Build only MLIRTest workloads
bbdev workload --build "--chip pebble --mlirtest"
```

**Response**:
```json
{
  "status": 200,
  "body": {
    "success": true,
    "processing": false,
    "return_code": 0
  }
}
```

## Notes

- Workload build directory is `bb-tests/workloads/build/<chip>`
- Workload source code is distributed under `bb-tests/workloads/src` and `examples/*/*/workloads`
- Workload binaries are emitted under `bb-tests/output/<chip>/workloads/src`
