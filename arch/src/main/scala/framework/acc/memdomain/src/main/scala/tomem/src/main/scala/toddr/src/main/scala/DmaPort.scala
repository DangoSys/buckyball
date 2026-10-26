package framework.acc.memdomain.toddr

import framework.core.memory.access.preflight.configs.PmpParams

import framework.acc.memdomain.tomem.configs.{Params => ModuleParam}

import chisel3._
import framework.acc.memdomain.MemoryContext
import chisel3.util._
import framework.core.memory.access.preflight.Params

class ReadTransfer extends Bundle {
  val vaddr        = UInt(64.W)
  val len          = UInt(32.W)
  val stride       = UInt(19.W)
  val groups       = UInt(6.W)
  val is_2d        = Bool()
  val pixel_bytes  = UInt(10.W)
  val source_width = UInt(10.W)
  val core_width   = UInt(4.W)
}

class ReadResult(b: ModuleParam) extends Bundle {
  val dataWidth = b.axi.dataBits

  val data        = UInt(dataWidth.W)
  val last        = Bool()
  val addrcounter = UInt(16.W)
  val fault       = new DmaStatus
}

class WriteTransfer(b: ModuleParam, permission: PmpParams) extends Bundle {
  val vaddr  = UInt(64.W)
  val len    = UInt(32.W)
  val access = new AccessRange(b, permission)
}

class WriteData(b: ModuleParam) extends Bundle {
  val dataWidth = b.axi.dataBits

  val data = UInt(dataWidth.W)
  val last = Bool()
}

class WriteResult extends Bundle {
  val done  = Bool()
  val fault = new DmaStatus
}

class DmaDecision(b: ModuleParam) extends Bundle {
  val p = b.mapping

  val valid    = Bool()
  val parentId = UInt(p.idBits.W)
  val fault    = new DmaStatus
}

class AccessRange(b: ModuleParam, permission: PmpParams) extends Bundle {
  val context      = new MemoryContext(permission)
  val baseVA       = UInt(64.W)
  val rows         = UInt(64.W)
  val columns      = UInt(64.W)
  val spanBytes    = UInt(64.W)
  val columnStride = UInt(64.W)
  val rowStride    = UInt(64.W)
}

class DmaReadCommand(b: ModuleParam, permission: PmpParams) extends Bundle {
  val transfer = new ReadTransfer
  val producer = UInt(2.W)
  val access   = new AccessRange(b, permission)
}

/** Accelerator-side logical transfers. Translation and external-memory transport belong to the system. */
class DmaPort(b: ModuleParam, permission: PmpParams) extends Bundle {
  val dataBits = b.axi.dataBits

  val read        = Decoupled(new DmaReadCommand(b, permission))
  val readResult  = Flipped(Decoupled(new ReadResult(b)))
  val write       = Decoupled(new WriteTransfer(b, permission))
  val writeData   = Decoupled(new WriteData(b))
  val writeResult = Flipped(Decoupled(new WriteResult))
  val readBusy    = Input(Bool())
  val writeBusy   = Input(Bool())
}
