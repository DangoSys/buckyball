# DMA Engine Implementation

## Overview

DMA engine implementation for Buckyball's memory domain, located at `arch/src/main/scala/framework/builtin/memdomain/dma`. Provides high-performance memory data transfer services between main memory and on-chip storage.

Main components:
- **BBStreamReader**: Streaming data reader for bulk reads from external memory
- **BBStreamWriter**: Streaming data writer for bulk writes to external memory
- **LocalAddr**: Local address management for Scratchpad and Accumulator mapping

## File Structure

```
dma/
├── DMA.scala         - Streaming DMA read/write implementation
└── LocalAddr.scala   - Local address management
```

## DMA.scala

### Request/Response Interfaces

```scala
class BBReadRequest()(implicit p: Parameters) extends CoreBundle {
  val vaddr = UInt(coreMaxAddrBits.W)  // Virtual address
  val len = UInt(16.W)                 // Read length (bytes)
  val status = new MStatus             // Processor status
}

class BBWriteRequest(dataWidth: Int)(implicit p: Parameters) extends CoreBundle {
  val vaddr = UInt(coreMaxAddrBits.W)      // Virtual address
  val data = UInt(dataWidth.W)             // Write data
  val len = UInt(16.W)                     // Write length (bytes)
  val mask = UInt((dataWidth / 8).W)       // Byte mask
  val status = new MStatus                 // Processor status
}
```

### BBStreamReader

**State Machine**:
```scala
val s_idle :: s_req_new_block :: Nil = Enum(2)
val state = RegInit(s_idle)
```

**Byte Counting**:
```scala
val bytesRequested = Reg(UInt(16.W))  // Bytes requested
val bytesReceived = Reg(UInt(16.W))   // Bytes received
val bytesLeft = req.len - bytesRequested
```

**TileLink Request**:
```scala
val get = edge.Get(
  fromSource = xactId,
  toAddress = 0.U,
  lgSize = log2Ceil(beatBytes).U
)._2
```

**TLB Integration**:
```scala
io.tlb.req.bits.tlb_req.vaddr := tlb_q.io.deq.bits.vaddr
io.tlb.req.bits.tlb_req.cmd := M_XRD  // Read operation
io.tlb.req.bits.status := tlb_q.io.deq.bits.status
```

### BBStreamWriter

**Put Operation Selection**:
```scala
val use_put_full = req.mask === ~0.U(beatBytes.W)
val selected_put = Mux(use_put_full, putFull, putPartial)
```

**Response Handling**:
```scala
io.resp.valid := tl.d.valid && edge.last(tl.d)
io.resp.bits.done := true.B
```

## LocalAddr.scala

### Address Structure

```scala
class LocalAddr(sp_banks: Int, sp_bank_entries: Int, acc_banks: Int, acc_bank_entries: Int) extends Bundle {
  val is_acc_addr = Bool()         // Is accumulator address
  val accumulate = Bool()          // Perform accumulation
  val read_full_acc_row = Bool()   // Read full accumulator row
  val data = UInt(memAddrBits.W)   // Actual address data
}
```

### Address Decomposition

```scala
// Scratchpad address decomposition
def sp_bank(dummy: Int = 0) = if (spAddrBits == spBankRowBits) 0.U
                             else data(spAddrBits - 1, spBankRowBits)
def sp_row(dummy: Int = 0) = data(spBankRowBits - 1, 0)

// Accumulator address decomposition
def acc_bank(dummy: Int = 0) = if (accAddrBits == accBankRowBits) 0.U
                              else data(accAddrBits - 1, accBankRowBits)
def acc_row(dummy: Int = 0) = data(accBankRowBits - 1, 0)
```

### Address Operations

```scala
// Address addition
def +(other: UInt) = {
  val result = WireInit(this)
  result.data := data + other
  result
}

// Addition with overflow check
def add_with_overflow(other: UInt): Tuple2[LocalAddr, Bool] = {
  val sum = data +& other
  val overflow = Mux(is_acc_addr, sum(accAddrBits), sum(spAddrBits))
  (result, overflow)
}
```

## Important Notes

1. **Alignment Requirements**: DMA operations consider TileLink protocol alignment requirements
2. **Transaction ID Management**: Both engines implement transaction ID allocation and recycling for concurrent requests
3. **TLB Integration**: Full virtual address translation support for user and kernel mode
4. **Pipeline Design**: Multiple pipeline stages including address translation, TileLink request, and response handling
5. **Error Handling**: TLB miss handling implemented, relies on upper layer software for access failures
6. **Performance**: BBStreamWriter supports full and partial write modes, automatically selects optimal TileLink operation based on mask
7. **Configuration**: DMA engines support parametrized configuration of concurrent transactions, data width, max transfer bytes
