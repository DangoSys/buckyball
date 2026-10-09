package framework.ant

import chisel3._
import chisel3.util._
import memcore.memory.spm

/** RV64IM local execution; no compressed, floating-point, atomic or privileged instructions. */
case class Params(
  codeBytes: Int,
  data:      spm.Params,
  shared:    spm.Params,
  taskBits:  Int = 32) {
  require(codeBytes >= 32 && isPow2(codeBytes))
  require(data.base >= codeBytes && shared.base >= codeBytes)
  require(data.base + data.bytes <= shared.base || shared.base + shared.bytes <= data.base)
  require(data.dataBits == shared.dataBits)
  require(taskBits > 0)
}

class Start(p: Params) extends Bundle {
  val task     = UInt(p.taskBits.W)
  val entry    = UInt(64.W)
  val codeEnd  = UInt(64.W)
  val argument = UInt(64.W)
  val stack    = UInt(64.W)
}

class Completion(p: Params) extends Bundle {
  val task      = UInt(p.taskBits.W)
  val value     = UInt(64.W)
  val cancelled = Bool()
}

class Command(p: Params) extends Bundle {
  val task        = UInt(p.taskBits.W)
  val pc          = UInt(64.W)
  val instruction = UInt(32.W)
  val rs1         = UInt(64.W)
  val rs2         = UInt(64.W)
}

class Response(p: Params) extends Bundle {
  val task  = UInt(p.taskBits.W)
  val rd    = UInt(5.W)
  val data  = UInt(64.W)
  val error = Bool()
}

class Retire extends Bundle {
  val pc          = UInt(64.W)
  val instruction = UInt(32.W)
  val rd          = UInt(5.W)
  val data        = UInt(64.W)
}

/** Core-side local execution and uncached management boundary. */
class LocalPort(p: Params) extends Bundle {
  val inUse          = Input(Bool())
  val start          = Flipped(Decoupled(new Start(p)))
  val result         = Decoupled(new Completion(p))
  val cancel         = Input(Bool())
  val sharedBusy     = Input(Bool())
  val sharedHostBusy = Input(Bool())
  val running        = Output(Bool())
  val storageBusy    = Output(Bool())
  val retired        = Output(Valid(new Retire))
  val code           = Flipped(new spm.Port(spm.Params(0, p.codeBytes, p.data.dataBits)))
  val data           = Flipped(new spm.Port(p.data))
  val shared         = new spm.Port(p.shared)
}
