# FuncSim Workflow

Functional simulation workflow in BuckyBall framework, providing fast functional verification environment.

## API Usage

### `build`
**Endpoint**: `POST /funcsim/build`

**Function**: Build functional simulator

**Parameters**: No specific parameters

**Example**:
```bash
bbdev funcsim --build
```

### `sim`
**Endpoint**: `POST /funcsim/sim`

**Function**: Run functional simulation

**Parameters**:
- **`binary`** - Binary file path to simulate
- **`ext`** - Extension parameters

**Examples**:
```bash
# Basic simulation
bbdev funcsim --sim "--binary /path/to/test.elf"

# With extension parameters
bbdev funcsim --sim "--binary /path/to/test.elf --ext additional_args"
```

## Typical Workflow

```bash
# 1. Build functional simulator
bbdev funcsim --build

# 2. Run simulation
bbdev funcsim --sim "--binary ${buckyball}/bb-tests/workloads/build/src/CTest/ctest_basic-baremetal"
```

## Notes

- Only provides functional-level simulation, no timing information
- Ensure binary file path is correct and accessible
