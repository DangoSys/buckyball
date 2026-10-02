package sims.p2e

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.Instantiate
import _root_.circt.stage.ChiselStage
import framework.system.configloader.{ChipLoader, RocketTileCore}
import framework.memdomain.backend.privatepath.PrivateMemBackend
import framework.memdomain.backend.shared.SharedMemBackend
import framework.top.GlobalConfig
import framework.top.configs.SimParam

// Functional test harness, never part of a P2E workload or performance measurement.
class AccessBackendTest(base: GlobalConfig) extends Module {

  val b = base.copy(
    sim = SimParam(accessTest = true),
    memDomain = base.memDomain.copy(
      bankNum = 2,
      bankEntries = 16,
      bankChannel = 2,
      virtualBankCount = 64,
      sharedEnable = true,
      sharedBankNum = 2,
      sharedEntries = 16,
      sharedInputChannels = 2,
      nCores = 2
    )
  )

  val io = IO(new Bundle {
    val allocate      = Input(Bool())
    val second        = Input(Bool())
    val valid         = Input(UInt(4.W))
    val responseReady = Input(Bool())
    val inst          = Input(UInt(64.W))
    val mask          = Input(UInt(16.W))
    val data          = Input(UInt(128.W))
    val accepted      = Output(UInt(4.W))
  })

  val priv   = Instantiate(new PrivateMemBackend(b))
  val shared = Instantiate(new SharedMemBackend(b))
  priv.io.query_vbank_id             := 0.U
  shared.io.query_valid.foreach(_    := false.B)
  shared.io.query_hart_id.foreach(_  := 0.U)
  shared.io.query_vbank_id.foreach(_ := 0.U)
  for ((config, isShared) <- Seq((priv.io.config, false), (shared.io.config, true))) {
    config.valid          := io.allocate
    config.bits.vbank_id  := (if (isShared) 32.U else 2.U) + io.second
    config.bits.hart_id   := 4.U + (if (isShared) io.second else false.B)
    config.bits.group_id  := 0.U
    config.bits.is_multi  := false.B
    config.bits.alloc     := true.B
    config.bits.is_shared := isShared.B
    when(io.allocate)(assert(config.ready))
  }
  val ports = priv.io.mem_req.toSeq ++ shared.io.mem_req.toSeq
  for ((port, i)          <- ports.zipWithIndex) {
    port.bank_id             := (if (i < 2) 2 + i else 32 + i - 2).U
    port.group_id            := 0.U
    port.hart_id             := (if (i < 2) 4 else 4 + i - 2).U
    port.rob_id              := 1.U
    port.inst_id             := io.inst
    port.is_shared           := (i >= 2).B
    port.read.req.valid      := false.B
    port.read.req.bits.addr  := 0.U
    port.read.resp.ready     := true.B
    port.write.req.valid     := io.valid(i)
    port.write.req.bits.addr := 5.U
    port.write.req.bits.data := io.data
    port.write.req.bits.mask := io.mask.asTypeOf(port.write.req.bits.mask)
    port.write.resp.ready    := io.responseReady
  }
  io.accepted := VecInit(ports.map(_.write.req.fire)).asUInt
}

object ElaborateAccessBackendTest extends App {
  val topology = ChipLoader.load("../examples/chips/toy/configs/generated/chip.pb")
  val b        = topology.tiles.head.cores.head.asInstanceOf[RocketTileCore].buckyball.get
  ChiselStage.emitSystemVerilogFile(new AccessBackendTest(b), args = Array("--target-dir", args(0), "--split-verilog"))
}
