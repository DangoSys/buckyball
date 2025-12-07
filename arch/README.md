# Buckyball

## 1. Why Buckyball

> ❗Buckyball is not a specific NPU design, but a design framework. You can find various NPU design examples in the examples folder.


## 2. Architecture Overview

### 2.1 System Architecture

### 2.2 ISA Specification

#### 2.2.1 bb_mvin

**Function**: Load data from main memory to scratchpad memory

**func7**: `0010000` (24)

**Format**: `bb_mvin rs1, rs2`

**Operands**:
- `rs1`: Main memory address
- `rs2[spAddrLen-1:0]`: Scratchpad address
- `rs2[spAddrLen+9:spAddrLen]`: Number of rows (iteration count)

**Operation**: Load data from main memory address `rs1` to scratchpad address specified in `rs2`, with iteration count determining number of rows

rs1 format:
```
┌─────────────────────────────────────────────────────────────────┐
│                        mem_addr                                 │
│                    (memAddrLen bits)                            │
├─────────────────────────────────────────────────────────────────┤
│                    [memAddrLen-1:0]                             │
└─────────────────────────────────────────────────────────────────┘
```

rs2 format:
```
┌──────────────────────────────────┬──────────────────────────────────────────┐
│        row (iter)                │                sp_addr                   │
│     (10 bits)                    │            (spAddrLen bits)              │
├──────────────────────────────────┼──────────────────────────────────────────┤
│ [spAddrLen+9:spAddrLen]          │            [spAddrLen-1:0]               │
└──────────────────────────────────┴──────────────────────────────────────────┘
```

#### 2.2.2 bb_mvout

**Function**: Store data from scratchpad memory to main memory

**func7**: `0010001` (25)

**Format**: `bb_mvout rs1, rs2`

**Operands**:
- `rs1`: Main memory address
- `rs2[spAddrLen-1:0]`: Scratchpad address
- `rs2[spAddrLen+9:spAddrLen]`: Number of rows to store (iteration count)

**Operation**: Store data from scratchpad address specified in `rs2` to main memory address `rs1`

rs1 format:
```
┌─────────────────────────────────────────────────────────────────┐
│                        mem_addr                                 │
│                    (memAddrLen bits)                            │
├─────────────────────────────────────────────────────────────────┤
│                    [memAddrLen-1:0]                             │
└─────────────────────────────────────────────────────────────────┘
```

rs2 format:
```
┌──────────────────────────────────┬──────────────────────────────────────────┐
│        row (iter)                │                sp_addr                   │
│     (10 bits)                    │            (spAddrLen bits)              │
├──────────────────────────────────┼──────────────────────────────────────────┤
│   [spAddrLen+9:spAddrLen]        │             [spAddrLen-1:0]              │
└──────────────────────────────────┴──────────────────────────────────────────┘
```

#### 2.2.3 Ball Execution Instructions

**Function**: Execute computation on Ball devices (matrix multiply, transpose, im2col, ReLU, etc.)

**func7**: `0100000` - `0111111` (32-63)

**Format**: `bb_<op> rs1, rs2`

**Common Operands**:
- `rs1[spAddrLen-1:0]`: First operand scratchpad address
- `rs1[2*spAddrLen-1:spAddrLen]`: Second operand scratchpad address
- `rs2[spAddrLen-1:0]`: Result write-back scratchpad address
- `rs2[spAddrLen+9:spAddrLen]`: Iteration count

rs1 format:
```
┌────────────────────────────────┬──────────────────────────────┐
│           op2_spaddr           │          op1_spaddr          │
│       (spAddrLen bits)         │      (spAddrLen bits)        │
├────────────────────────────────┼──────────────────────────────┤
│  [2*spAddrLen-1:spAddrLen]     │ [spAddrLen-1:0]              │
└────────────────────────────────┴──────────────────────────────┘
```

rs2 format:
```
┌──────────────────────────┬────────────────────────────────────┐
│         iter             │    wr_spaddr                       │
│       (10 bits)          │  (spAddrLen bits)                  │
├──────────────────────────┼────────────────────────────────────┤
│ [spAddrLen+9:spAddrLen]  │  [spAddrLen-1:0]                   │
└──────────────────────────┴────────────────────────────────────┘
```
