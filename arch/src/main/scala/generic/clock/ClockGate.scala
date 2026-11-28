package framework.generic.clock

import chisel3._
import chisel3.util.HasBlackBoxInline
import chisel3.experimental.hierarchy.{instantiable, public}

/**
 * The clock of a unit while `enable` is set: the enable is sampled while the clock is low, so a
 * unit enabled in a cycle takes that cycle's rising edge and none is cut short. A synthesis flow
 * maps it to the library's clock gate.
 */
@instantiable
class ClockGate extends BlackBox with HasBlackBoxInline {

  @public
  val io = IO(new Bundle {
    val clock  = Input(Clock())
    val enable = Input(Bool())
    val gated  = Output(Clock())
  })

  setInline(
    "ClockGate.sv",
    """
      |module ClockGate (
      |  input  clock,
      |  input  enable,
      |  output gated
      |);
      |  reg sampled;
      |  always @(negedge clock) sampled <= enable;
      |  assign gated = clock & sampled;
      |endmodule
      |""".stripMargin
  )
}
