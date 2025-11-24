# UVM Workflow

UVM (Universal Verification Methodology) workflow in the Buckyball framework for building and running UVM verification environments.

## API Usage

### `builddut`
**Endpoint**: `POST /uvm/builddut`

**Function**: Build DUT (Design Under Test)

**Parameters**:
- **`jobs`** - Number of parallel build tasks, default 16

**Example**:
```bash
# Build DUT with default parallelism
bbdev uvm --builddut

# Specify number of parallel tasks
bbdev uvm --builddut "--jobs 8"
```

### `build`
**Endpoint**: `POST /uvm/build`

**Function**: Build UVM executable

**Parameters**:
- **`jobs`** - Number of parallel build tasks, default 16

**Example**:
```bash
# Build UVM with default parallelism
bbdev uvm --build

# Specify number of parallel tasks
bbdev uvm --build "--jobs 8"
```

## Typical Workflow

```bash
# 1. Build DUT
bbdev uvm --builddut

# 2. Build UVM environment
bbdev uvm --build
```

**Response Format**:
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
