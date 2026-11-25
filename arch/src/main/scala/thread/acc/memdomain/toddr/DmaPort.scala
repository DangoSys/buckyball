package framework.thread.acc.memdomain.toddr

import framework.generic.contract.memory.access.preflight.Params
import framework.thread.acc.memdomain.toddr.AccessRange

import framework.thread.acc.memdomain.tomem.configs.{Params => ModuleParam}
import framework.thread.acc.memdomain.toddr.DmaPort
import framework.thread.acc.memdomain.toddr.DmaReadCommand
import framework.thread.acc.memdomain.toddr.ReadResult
import framework.thread.acc.memdomain.toddr.ReadTransfer
import framework.thread.acc.memdomain.toddr.WriteData
import framework.thread.acc.memdomain.toddr.WriteResult
import framework.thread.acc.memdomain.toddr.WriteTransfer

import framework.generic.contract.memdomain.MemoryContext
import chisel3._
import chisel3.util._
import framework.generic.contract.memory.access.preflight.configs.PmpParams
import framework.generic.contract.memdomain.toddr.DmaStatus

class ReadTransfer extends Bundle {
  val vaddr        = UInt(64.W)
  val len          = UInt(32.W)
  val stride       = UInt(19.W)
  val groups       = UInt(6.W)
  val is_2d        = Bool()
  val is_strided   = Bool()
  val byte_stride  = UInt(20.W)
  val row_bytes    = UInt(5.W)
  val block_log2   = UInt(3.W)
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

class AccessRange(b: ModuleParam, permission: PmpParams) extends Bundle {
  val context      = new MemoryContext(permission)
  val baseVA       = UInt(64.W)
  val rows         = UInt(64.W)
  val columns      = UInt(64.W)
  val spanBytes    = UInt(64.W)
  val columnStride = UInt(64.W)
  val rowStride    = UInt(64.W)
}
