# Workload Workflow

Workload build workflow in BuckyBall framework, used to build test workloads and benchmark programs.

## API Usage

### `build`
**Endpoint**: `POST /workload/build`

**Function**: Build workload

**Parameters**:
- **`workload`** - Specify workload name to build

**Examples**:
```bash
# Build specific workload
bbdev workload --build "--workload test_program"

# Build all workloads
bbdev workload --build
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

- Workload source code located in `bb-tests/workload` directory
- Build results typically output to `bb-tests/workloads/build` directory
