package framework.rvv

import chisel3._
import chisel3.util._
import framework.balldomain.blink.BlinkPortIO
import framework.top.GlobalConfig

class KernelBlinkIO(b: GlobalConfig)
    extends BlinkPortIO(b, b.rvv.memoryPorts, b.rvv.memoryPorts, new KernelRequest(b), new KernelResponse(b)) {
  // Images contain instructions and read-only constants; mutable globals are
  // rejected. Temporary state stays inside the kernel; data outputs use banks.
  val image         = Flipped(Decoupled(UInt(32.W)))
  val imageTerminal = Flipped(Decoupled(new ImageTerminal))
  // Sample diagnostics with cmdResp; no separate completion handshake.
  val result        = Output(new KernelCompletion)
}

class KernelCommand extends Bundle {
  val bid    = UInt(5.W)
  val funct7 = UInt(7.W)
  val rs1    = UInt(64.W)
  val rs2    = UInt(64.W)
}

class KernelRequest(b: GlobalConfig) extends Bundle {
  val cmd    = new KernelCommand
  val rob_id = UInt(log2Up(b.frontend.rob_entries).W)
}

class KernelResponse(b: GlobalConfig) extends Bundle {
  val rob_id     = UInt(log2Up(b.frontend.rob_entries).W)
  val write_bank = UInt(16.W)
}

class ImageTerminal extends Bundle {
  val error   = UInt(4.W)
  val address = UInt(64.W)
}
