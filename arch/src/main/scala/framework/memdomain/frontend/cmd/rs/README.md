# Reservation Station (RS) Module

## Overview

Reservation station module for memory domain instruction scheduling, located at `framework/memdomain/frontend/cmd/rs`. Local Mem RS is a small FIFO scheduler; global reorder lives in Frontend `GlobalROB`.

## File Structure

```
rs/
└── reservationStation.scala  - Memory reservation station
```

## MemReservationStation

### Interface

```scala
class MemReservationStation(implicit b: CustomBuckyballConfig, p: Parameters) extends Module {
  val io = IO(new Bundle {
    val mem_decode_cmd_i = Flipped(Decoupled(new MemDecodeCmd))
    val rs_rocc_o = new Bundle {
      val resp  = Decoupled(new RoCCResponseBB)
      val busy  = Output(Bool())
    }
    val issue_o     = new MemIssueInterface
    val commit_i    = new MemCommitInterface
  })
}
```

### Issue Interface

```scala
class MemIssueInterface(implicit b: CustomBuckyballConfig, p: Parameters) extends Bundle {
  val ld = Decoupled(new MemRsIssue)    // Load instruction issue
  val st = Decoupled(new MemRsIssue)    // Store instruction issue
}
```

### Commit Interface

```scala
class MemCommitInterface(implicit b: CustomBuckyballConfig, p: Parameters) extends Bundle {
  val ld = Flipped(Decoupled(new MemRsComplete))    // Load completion
  val st = Flipped(Decoupled(new MemRsComplete))    // Store completion
}
```

## Related Modules

- [Memory Domain](../README.md)
- [Memory Controller](../mem/README.md)
- [DMA Engines](../dma/README.md)
