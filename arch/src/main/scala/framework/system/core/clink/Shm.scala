package framework.system.core.clink

import chisel3._
import chisel3.util._
import framework.top.GlobalConfig
import framework.memdomain.backend.MemRequestIO
import framework.memdomain.backend.shared.SharedMemLayout
import framework.memdomain.frontend.mem.MemConfigerIO
import framework.memdomain.isa.{MvoverISA, MvoverPort}
import memcore.memory.mesh_shm.MeshLocalBankPort

/** Core-facing access to the tile's shared bank network. */
class ShmPort(b: GlobalConfig) extends Bundle {
  val requests = Vec(SharedMemLayout.channelPerHart(b), new MemRequestIO(b))
  val move     = new MvoverPort

  val local = Flipped(new MeshLocalBankPort(
    MvoverISA.AddressBits,
    MvoverISA.BankBits,
    b.memDomain.bankWidth,
    math.max(1, log2Ceil(b.frontend.rob_entries))
  ))

  val config         = Decoupled(new MemConfigerIO(b))
  val queryValid     = Output(Bool())
  val queryVbank     = Output(UInt(b.memDomain.vbankIdWidth.W))
  val queryGroups    = Input(UInt(b.memDomain.groupCountWidth.W))
  val barrierArrive  = Output(Bool())
  val barrierRelease = Input(Bool())
}
