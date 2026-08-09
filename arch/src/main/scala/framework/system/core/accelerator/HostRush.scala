package framework.system.core.accelerator

import chisel3._
import chisel3.util.{Decoupled, HasBlackBoxInline}
import org.chipsalliance.cde.config.Field
import framework.system.core.rocket.RoCCCommandBB

/** Enables the Verilator-only host command source at elaboration time. */
case object BuckyballHostRushKey extends Field[Boolean](false)

/** Host ABI ID: tile ID in the high half, local accelerator index below it. */
object HostRushAcceleratorId {
  private val LocalIdBits = 16
  private val LocalIdMask = (1 << LocalIdBits) - 1

  def apply(tileId: Int, localIndex: Int): Int = {
    require(tileId >= 0 && tileId < (1 << LocalIdBits), s"tile ID does not fit host-rush ABI: $tileId")
    require(localIndex >= 0 && localIndex <= LocalIdMask, s"accelerator index does not fit host-rush ABI: $localIndex")
    (tileId << LocalIdBits) | localIndex
  }

}

/**
 * Stable DPI boundary for host-driven RTL simulation.
 *
 * One instance is created for every Buckyball accelerator. The accelerator ID
 * is an ABI identifier, not a hart ID: heterogeneous systems may assign
 * arbitrary hart IDs and may give individual accelerators different configs.
 */
class HostRushCommandDPI(acceleratorId: Int, xLen: Int)
    extends BlackBox(Map(
      "ACCELERATOR_ID" -> acceleratorId,
      "XLEN"           -> xLen
    ))
    with HasBlackBoxInline {

  val io = IO(new Bundle {
    val clock   = Input(Clock())
    val ready   = Input(Bool())
    val retired = Input(Bool())
    val valid   = Output(Bool())
    val funct   = Output(UInt(7.W))
    val rs1Data = Output(UInt(xLen.W))
    val rs2Data = Output(UInt(xLen.W))
  })

  setInline(
    "HostRushCommandDPI.v",
    """
      |module HostRushCommandDPI #(
      |  parameter integer ACCELERATOR_ID = 0,
      |  parameter integer XLEN = 64
      |)(
      |  input clock, input ready, input retired,
      |  output bit valid,
      |  output logic [6:0] funct,
      |  output logic [XLEN-1:0] rs1Data, output logic [XLEN-1:0] rs2Data
      |);
      |  import "DPI-C" function void verilator_host_rush_peek(
      |    input int accelerator_id,
      |    output bit valid,
      |    output longint unsigned xs1_data,
      |    output longint unsigned xs2_data,
      |    output int unsigned funct);
      |  import "DPI-C" function void verilator_host_rush_accept(input int accelerator_id);
      |  import "DPI-C" function void verilator_host_rush_observe(
      |    input int accelerator_id, input bit valid, input bit ready);
      |  import "DPI-C" function void verilator_host_rush_report(
      |    input int accelerator_id, input bit retired);
      |
      |  bit accept_pending = 1'b0;
      |  int unsigned dpi_funct;
      |
      |  always @(posedge clock) begin
      |    verilator_host_rush_observe(ACCELERATOR_ID, valid, ready);
      |    accept_pending <= valid && ready;
      |    verilator_host_rush_report(ACCELERATOR_ID, retired);
      |  end
      |
      |  always @(negedge clock) begin
      |    if (accept_pending) begin
      |      verilator_host_rush_accept(ACCELERATOR_ID);
      |      accept_pending <= 1'b0;
      |      valid = 1'b0;
      |    end else if (!valid) begin
      |      // DPI calls mutate C++ state without creating an RTL event. Load a
      |      // one-entry register on the falling edge, so command bits are
      |      // stable for the full following sampling edge.
      |      verilator_host_rush_peek(ACCELERATOR_ID, valid, rs1Data, rs2Data, dpi_funct);
      |      funct = dpi_funct[6:0];
      |    end
      |  end
      |
      |  initial begin
      |    valid = 1'b0;
      |    rs1Data = '0;
      |    rs2Data = '0;
      |    funct = '0;
      |  end
      |endmodule
      |""".stripMargin
  )
}

class HostRushCommandBridge(acceleratorId: Int, xLen: Int) extends Module {

  val io = IO(new Bundle {
    val cmd     = Decoupled(new RoCCCommandBB(xLen))
    val retired = Input(Bool())
  })

  val dpi = Module(new HostRushCommandDPI(acceleratorId, xLen))
  dpi.io.clock   := clock
  dpi.io.ready   := io.cmd.ready
  dpi.io.retired := io.retired

  io.cmd.valid         := dpi.io.valid
  io.cmd.bits.raw_inst := 0.U
  io.cmd.bits.pc       := 0.U
  io.cmd.bits.funct    := dpi.io.funct
  io.cmd.bits.funct3   := "b011".U
  io.cmd.bits.rs2      := 0.U
  io.cmd.bits.rs1      := 0.U
  io.cmd.bits.xd       := false.B
  io.cmd.bits.xs1      := true.B
  io.cmd.bits.xs2      := true.B
  io.cmd.bits.rd       := 0.U
  io.cmd.bits.opcode   := "h7b".U
  io.cmd.bits.rs1Data  := dpi.io.rs1Data
  io.cmd.bits.rs2Data  := dpi.io.rs2Data
}
