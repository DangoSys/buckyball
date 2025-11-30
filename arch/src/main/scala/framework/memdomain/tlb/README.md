# TLB Module (Translation Lookaside Buffer)

## Overview

The TLB module implements address translation caching from virtual to physical addresses, located at `framework/builtin/memdomain/tlb`. Based on Rocket-chip's TLB implementation, it provides Buckyball-specific TLB encapsulation and cluster management.

## File Structure

```
tlb/
├── BBTLB.scala           - Buckyball TLB implementation
├── TLBCluster.scala      - TLB cluster manager
├── spec-BBTLB.md         - BBTLB specification
└── spec-BBTLBCluster.md  - TLB cluster specification
```

## Core Components

### BBTLB - Buckyball TLB

BBTLB wraps Rocket-chip TLB with Buckyball-specific interface and exception handling:

```scala
class BBTLB(entries: Int, maxSize: Int)(implicit edge: TLEdgeOut, p: Parameters)
  extends CoreModule {

  val lgMaxSize = log2Ceil(maxSize)
  val io = IO(new Bundle {
    val req = Flipped(Valid(new BBTLBReq(lgMaxSize)))
    val resp = new TLBResp
    val ptw = new TLBPTWIO
    val exp = new BBTLBExceptionIO
  })
}
```

**Request Interface**:
```scala
class BBTLBReq(val lgMaxSize: Int)(implicit p: Parameters) extends CoreBundle {
  val tlb_req = new TLBReq(lgMaxSize)    // TLB request
  val status = new MStatus               // Processor status
}
```

**Exception Interface**:
```scala
class BBTLBExceptionIO extends Bundle {
  val interrupt = Output(Bool())         // Interrupt output
  val flush_retry = Input(Bool())        // Retry flush
  val flush_skip = Input(Bool())         // Skip flush

  def flush(dummy: Int = 0): Bool = flush_retry || flush_skip
}
```

**Implementation**: Internally instantiates Rocket-chip TLB with single-set configuration:
```scala
val tlb = Module(new TLB(false, lgMaxSize, TLBConfig(nSets=1, nWays=entries)))
```

**Exception Detection**:
```scala
val exception = io.req.valid && Mux(
  io.req.bits.tlb_req.cmd === M_XRD,
  tlb.io.resp.pf.ld || tlb.io.resp.ae.ld,  // Read exceptions
  tlb.io.resp.pf.st || tlb.io.resp.ae.st   // Write exceptions
)
```

### BBTLBCluster - TLB Cluster

BBTLBCluster manages multiple TLB instances for concurrent client access:

```scala
class BBTLBCluster(nClients: Int, entries: Int, maxSize: Int)
                 (implicit edge: TLEdgeOut, p: Parameters) extends CoreModule {

  val io = IO(new Bundle {
    val clients = Flipped(Vec(nClients, new BBTLBIO))
    val ptw = Vec(nClients, new TLBPTWIO)
    val exp = Vec(nClients, new BBTLBExceptionIO)
  })
}
```

**Client Interface**:
```scala
class BBTLBIO(implicit p: Parameters) extends CoreBundle {
  val lgMaxSize = log2Ceil(coreDataBytes)
  val req = Valid(new BBTLBReq(lgMaxSize))    // TLB request
  val resp = Flipped(new TLBResp)             // TLB response
}
```

**L0 TLB Cache**: Each client has an L0 TLB cache for recent translations:
```scala
val last_translated_valid = RegInit(false.B)
val last_translated_vpn = RegInit(0.U(vaddrBits.W))
val last_translated_ppn = RegInit(0.U(paddrBits.W))

val l0_tlb_hit = last_translated_valid &&
  ((client.req.bits.tlb_req.vaddr >> pgIdxBits).asUInt ===
   (last_translated_vpn >> pgIdxBits).asUInt)
```

**Translation Flow**:
1. **L0 Cache Check**: First check L0 TLB cache for hit
2. **L1 TLB Query**: Query L1 TLB on L0 miss
3. **Page Table Walk**: PTW on TLB miss
4. **Cache Update**: Update L0 cache on successful translation

```scala
when (tlbReqFire && !tlb.io.resp.miss) {
  last_translated_valid := true.B
  last_translated_vpn := tlbReq.tlb_req.vaddr
  last_translated_ppn := tlb.io.resp.paddr
}
```

## Configuration

**TLB Parameters**:
- `entries`: Number of TLB entries
- `maxSize`: Maximum transfer size
- `nClients`: Number of clients

**Example**:
```scala
val tlbConfig = TLBConfig(
  nSets = 1,           // TLB sets
  nWays = 32           // Ways per set
)
```

## Usage

### Single TLB

```scala
val bbtlb = Module(new BBTLB(entries = 32, maxSize = 64))

// Connect request
bbtlb.io.req.valid := tlbReqValid
bbtlb.io.req.bits.tlb_req := tlbRequest
bbtlb.io.req.bits.status := processorStatus

// Get response
val tlbResp = bbtlb.io.resp
val physicalAddr = tlbResp.paddr
val tlbMiss = tlbResp.miss

// Connect PTW
ptw <> bbtlb.io.ptw

// Handle exceptions
when(bbtlb.io.exp.interrupt) {
  // Handle TLB exception
}
```

### TLB Cluster

```scala
val tlbCluster = Module(new BBTLBCluster(
  nClients = 4,
  entries = 32,
  maxSize = 64
))

// Connect clients
for (i <- 0 until nClients) {
  tlbCluster.io.clients(i).req.valid := clientReqValid(i)
  tlbCluster.io.clients(i).req.bits := clientReq(i)
  clientResp(i) := tlbCluster.io.clients(i).resp
}

// Connect PTW
for (i <- 0 until nClients) {
  ptw(i) <> tlbCluster.io.ptw(i)
}
```

## Address Translation

### Virtual Address Format

```
Virtual Address (64-bit):
[63:39] [38:30] [29:21] [20:12] [11:0]
  VPN3    VPN2    VPN1    VPN0   Offset
```

### Physical Address Format

```
Physical Address:
[PPN][Offset]
```

### Translation Process

1. **Address Parsing**: Parse VPN and offset from virtual address
2. **TLB Lookup**: Look up PPN for VPN in TLB
3. **Page Table Walk**: PTW on TLB miss
4. **Permission Check**: Check access permissions
5. **Address Composition**: Combine PPN and offset to form physical address

## Exception Handling

### Exception Types

- **Page Fault**: Access to invalid page
- **Access Exception**: Insufficient permissions
- **TLB Miss**: Requires page table walk

### Exception Handling Flow

```scala
val exception = io.req.valid && Mux(
  io.req.bits.tlb_req.cmd === M_XRD,
  tlb.io.resp.pf.ld || tlb.io.resp.ae.ld,    // Read exception
  tlb.io.resp.pf.st || tlb.io.resp.ae.st     // Write exception
)
```

### TLB Flush

```scala
tlb.io.sfence.valid := io.exp.flush()
tlb.io.sfence.bits.rs1 := false.B
tlb.io.sfence.bits.rs2 := false.B
```

## Performance Optimization

### L0 TLB Cache

- Reduces L1 TLB access latency
- Improves address translation throughput
- Lowers power consumption

### Parallel Processing

- Multiple clients access in parallel
- Independent page table walkers
- Separate exception handling

## Related Modules

- [Memory Domain Overview](../README.md)
- [DMA Engines](../dma/README.md)
- [Memory Controller](../mem/README.md)
