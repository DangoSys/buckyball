# Workload Workflow

Workload build workflow in Buckyball framework, used to build test workloads and benchmark programs.

## API Usage

### `clean`
**Endpoint**: `POST /workload/clean`

**Function**: Clean the workloads of one selection: deletes `output/sims/<selection>/workloads` and `output/sims/<selection>/workloads-build`.

**Parameters**:
- **`config`** - Required. Config TOML path relative to `examples/designs`; the design is its first path segment.
- **`device`** / **`hart`** - Exactly one is required; selects the component of the config.

**Examples**:
```bash
bbdev workload --clean "--device toy"
```

### `build`
**Endpoint**: `POST /workload/build`

**Function**: Build workload. Builds the stack of the simulation (design.cmake),
then the C API `bb-tests/workloads/lib` (bbsw, its own CMake build in
`output/sims/<selection>/bbsw`), then the workloads with both.

**Parameters**:
- **`config`** - Required. Config TOML path relative to `examples/designs`; the design is its first path segment.
- **`device`** / **`hart`** - Exactly one is required; selects the component of the config.
- **`ctest`** - Build CTest workloads only.
- **`mlirtest`** - Build MLIRTest workloads only.
- **`soctest`** - Build SoC test workloads only.

`ctest`, `mlirtest` and `soctest` are mutually exclusive. Products are copied to `output/sims/<selection>/workloads`.

**Examples**:
```bash
# Build all workloads
bbdev workload --build "--device toy"

# Build only CTest workloads
bbdev workload --build "--device pebble --ctest"

# Build only MLIRTest workloads
bbdev workload --build "--device pebble --mlirtest"

# Build only SoC test workloads
bbdev workload --build "--device goban --soctest"
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

- `output/sims/<selection>` is `examples/designs/<design>/output/sims/<level>/<name>/<Config>`
- Workload build directory is `output/sims/<selection>/workloads-build`
- Workload source code is distributed under `bb-tests/workloads/src` and `examples/*/*/workloads`
- Workload binaries are emitted under `output/sims/<selection>/workloads`
