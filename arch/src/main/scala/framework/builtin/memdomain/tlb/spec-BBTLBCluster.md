# BBTLBCluster (Translation Lookaside Buffer Cluster) Specification

## Overview

BBTLBCluster is a multi-client TLB cluster implementation that supports concurrent virtual address translation for multiple clients. Inheriting from CoreModule, it instantiates multiple BBTLB modules to provide independent TLB services for each client, while implementing an L0-level fast translation cache mechanism to improve performance. The module supports parameterized configuration of client count, TLB entries, and maximum page size.

## Interface Design

BBTLBCluster's IO interface contains three main components:
- Client interface (clients): Provides Vec(nClients, BBTLBIO), each client has independent request and response channels
- Page table walker interface (ptw): Vec(nClients, TLBPTWIO), provides independent PTW connection for each client
- Exception handling interface (exp): Vec(nClients, BBTLBExceptionIO), manages exceptions and flush operations for each client

BBTLBIO interface definition:
- req: Valid(BBTLBReq), contains TLB request and related status information
- resp: Flipped(TLBResp), returns address translation result, miss flag, and exception information

## Internal Architecture

### TLB Instantiation
The module instantiates an independent BBTLB module for each client through `Seq.fill(nClients)(Module(new BBTLB(entries, maxSize)))` to create a TLB array. Each TLB instance's PTW and exception interfaces are directly connected to corresponding output ports.

### L0 Fast Cache Mechanism
To improve translation performance for frequently accessed addresses, each client implements a single-entry L0-level translation cache:
- `last_translated_valid`: Indicates cache entry validity
- `last_translated_vpn`: Cached virtual page number
- `last_translated_ppn`: Cached physical page number

L0 cache hit condition: cache is valid and current request's virtual page number matches cached virtual page number.

## Address Translation Flow

### L0 Cache Lookup
Each client's address translation first looks up in the L0 cache:
1. Check `l0_tlb_hit` condition: cache valid and page number match
2. If hit, directly calculate physical address: `Cat(last_translated_ppn >> pgIdxBits, vaddr(pgIdxBits-1,0))`
3. If miss, forward request to corresponding BBTLB module

### TLB Lookup Flow
When L0 cache misses:
1. Use `RegNext` to delay one cycle and forward client request to TLB
2. TLB valid signal set to: `RegNext(client.req.valid && !l0_tlb_hit)`
3. TLB request data set to: `RegNext(client.req.bits)`

### Cache Update Mechanism
When TLB request completes without miss, update L0 cache:
```scala
when (tlbReqFire && !tlb.io.resp.miss) {
  last_translated_valid := true.B
  last_translated_vpn := tlbReq.tlb_req.vaddr
  last_translated_ppn := tlb.io.resp.paddr
}
```

## Response Path Design

Module implements dual-path response mechanism:
1. **TLB Path**: When TLB request fires, directly return TLB response result
2. **L0 Cache Path**: When using L0 cache, return cached calculated physical address, miss flag set to `!RegNext(l0_tlb_hit)`

Register `l0_tlb_paddr_reg` saves client request's virtual address for L0 cache path response.

## Exception and Flush Handling

Module maintains L0 cache coherency by monitoring each TLB's flush signal:
```scala
when (tlb.io.exp.flush()) {
  last_translated_valid := false.B
}
```

When any flush operation occurs, L0 cache is immediately invalidated, ensuring address translation correctness.

## Parameterized Configuration

BBTLBCluster supports three main parameters:
- `nClients`: Number of supported clients, determines degree of concurrent TLB access
- `entries`: Number of entries per TLB instance
- `maxSize`: Maximum supported page size

`lgMaxSize` calculated through `log2Ceil(coreDataBytes)`, used to determine address width and related logic precision.

## Performance Optimization Features

1. **L0 Fast Cache**: Provides single-entry fast access for each client, reducing hot address access latency
2. **Parallel Processing**: Multiple clients can perform address translation simultaneously, improving system throughput
3. **Independent PTW**: Each client has independent page table walker interface, avoiding PTW resource contention
4. **Pipeline Design**: Uses register delays to implement pipeline operations, improving clock frequency
