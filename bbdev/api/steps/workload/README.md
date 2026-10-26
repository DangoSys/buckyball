# Workload Workflow

Workload build workflow in Buckyball framework, used to build test workloads and benchmark programs.

## API Usage

### `clean`
**Endpoint**: `POST /workload/clean`

**Function**: Clean workload output directory for one design.

**Parameters**:
- **`design`** - Required design name.

**Examples**:
```bash
bbdev workload --clean "--design toy"
```

### `build`
**Endpoint**: `POST /workload/build`

**Function**: Build workload

**Parameters**:
- **`design`** - Required design name. Selects design-specific workloads.
- **`stable`** - Optional boolean flag. If set, build with stable LLVM Buckyball extensions.
- **`ctest`** - Build CTest workloads only.
- **`mlirtest`** - Build MLIRTest workloads only.
- **`soctest`** - Build SoC test workloads only.

For design workloads under paths like `*/designs/<design>`, only the directory selected by `design` is synced to `bb-tests/output/<design>/workloads`.

**Examples**:
```bash
# Build all workloads
bbdev workload --build "--design toy"

# Build only CTest workloads
bbdev workload --build "--design pebble --ctest"

# Build only MLIRTest workloads
bbdev workload --build "--design pebble --mlirtest"

# Build only SoC test workloads
bbdev workload --build "--design goban --soctest"
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

- Workload build directory is `bb-tests/workloads/build/<design>`
- Workload source code is distributed under `bb-tests/workloads/src` and `examples/*/*/workloads`
- Workload binaries are emitted under `bb-tests/output/<design>/workloads/src`
