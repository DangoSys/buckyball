# FireSim Workflow

FireSim FPGA simulation workflow in Buckyball framework, providing FPGA-based hardware simulation environment.

## API Usage

### `buildbitstream`
**Endpoint**: `POST /firesim/buildbitstream`

**Function**: Build FPGA bitstream file

**Parameters**: No specific parameters

**Example**:
```bash
bbdev firesim --buildbitstream
```

### `infrasetup`
**Endpoint**: `POST /firesim/infrasetup`

**Function**: Setup FireSim infrastructure

**Parameters**: No specific parameters

**Example**:
```bash
bbdev firesim --infrasetup
```

### `runworkload`
**Endpoint**: `POST /firesim/runworkload`

**Function**: Run workload on FireSim

**Parameters**: No specific parameters

**Example**:
```bash
bbdev firesim --runworkload
```

## Typical Workflow

```bash
# 1. Build bitstream
bbdev firesim --buildbitstream

# 2. Setup infrastructure
bbdev firesim --infrasetup

# 3. Run workload
bbdev firesim --runworkload
```

## Notes

- Bitstream build takes several hours
- infrasetup requires cloud computing resource configuration
