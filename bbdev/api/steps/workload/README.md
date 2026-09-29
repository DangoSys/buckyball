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
- **`stable`** - Optional boolean flag. If set, build with stable LLVM Buckyball extensions.
- **`ctest`** - Build CTest workloads only.
- **`mlirtest`** - Build MLIRTest workloads only.

For chip workloads under paths like `*/chips/<chip>`, only the directory selected by `chip` is synced to `bb-tests/output/<chip>/workloads`.

**Examples**:
```bash
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
