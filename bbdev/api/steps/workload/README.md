# Workload Workflow

Workload build workflow in Buckyball framework, used to build test workloads and benchmark programs.

## API Usage

### `build`
**Endpoint**: `POST /workload/build`

**Function**: Build workload

**Parameters**:
- **`chip`** - Required chip name. Selects chip-specific workloads.
- **`model`** - Optional model name to build. If omitted, build all workloads.
- **`stable`** - Optional boolean flag. If set, build with stable LLVM Buckyball extensions.

**Examples**:
```bash
# Build one model with the default xbuckyball path
bbdev workload --build "--chip toy --model lenet"

# Build one model with stable LLVM Buckyball extensions
bbdev workload --build "--chip toy --model lenet --stable"

# Build all workloads
bbdev workload --build "--chip toy"
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

- Workload build entry is `bb-tests`
- Workload source code is distributed under `bb-tests/workloads/src` and `examples/*/*/workloads`
- Workload binaries are emitted under `bb-tests/output/workloads/src`
