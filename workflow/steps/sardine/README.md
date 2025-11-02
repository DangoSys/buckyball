# Sardine Workflow

Sardine workflow in the BuckyBall framework for running Sardine-related tasks.

## API Usage

### `run`
**Endpoint**: `POST /sardine/run`

**Function**: Run Sardine tasks

**Parameters**:
- **`workload`** - Specify the workload to run

**Example**:
```bash
# Run specified workload
bbdev sardine --run "--workload /path/to/workload"

# Run default workload
bbdev sardine --run
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
