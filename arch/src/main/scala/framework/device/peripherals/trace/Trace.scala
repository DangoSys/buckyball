package framework.device.peripherals.trace

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import framework.device.peripherals.DevicePort
import framework.device.peripherals.trace.configs.TraceParams

/** A simulation trace register block. The final word commits one binary event. */
@instantiable
class Trace(p: TraceParams) extends Module {

  @public
  val io = IO(new Bundle {
    val port = new DevicePort
  })

  val sink     = Instantiate(new TraceDPI)
  val access   = io.port.access.bits
  val offset   = access.addr - p.base.U
  val register = offset(log2Ceil(p.stride) - 1, 0)
  val valid    = access.addr >= p.base.U && access.addr < (p.base + p.bytes).U &&
    access.write && access.size === 3.U && register <= 88.U && register(2, 0) === 0.U
  io.port.read    := 0.U
  io.port.error   := !valid
  sink.io.clock   := clock
  sink.io.reset   := reset.asBool
  sink.io.enable  := io.port.access.valid && valid
  sink.io.channel := offset >> log2Ceil(p.stride)
  sink.io.word    := register >> 3
  sink.io.data    := access.data
}

@instantiable
class TraceDPI extends BlackBox with HasBlackBoxInline {

  @public
  val io = IO(new Bundle {
    val clock   = Input(Clock())
    val reset   = Input(Bool())
    val enable  = Input(Bool())
    val channel = Input(UInt(32.W))
    val word    = Input(UInt(32.W))
    val data    = Input(UInt(64.W))
  })

  setInline(
    "TraceDPI.v",
    """
      |module TraceDPI(
      |  input clock, reset, enable,
      |  input [31:0] channel, word,
      |  input [63:0] data
      |);
      |  import "DPI-C" context function void dpi_trace_write(
      |    input int unsigned channel, word, data_lo, data_hi);
      |  always @(posedge clock)
      |    if (!reset && enable)
      |      dpi_trace_write(channel, word, data[31:0], data[63:32]);
      |endmodule
      |""".stripMargin
  )
}
