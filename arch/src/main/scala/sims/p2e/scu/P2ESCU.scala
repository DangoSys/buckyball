package sims.p2e.scu

import chisel3._
import chisel3.util.HasBlackBoxInline
import org.chipsalliance.cde.config.{Config, Field, Parameters}

import freechips.rocketchip.devices.tilelink._
import freechips.rocketchip.diplomacy._
import freechips.rocketchip.regmapper._
import freechips.rocketchip.subsystem._
import freechips.rocketchip.tilelink._

case class P2ESCUParams(
  address: BigInt = BigInt("60000000", 16),
  size:    BigInt = BigInt("40000", 16))

case object P2ESCUKey extends Field[Option[P2ESCUParams]](None)

class WithP2ESCU(
  address: BigInt = BigInt("60000000", 16),
  size:    BigInt = BigInt("40000", 16))
    extends Config((site, here, up) => {
      case P2ESCUKey => Some(P2ESCUParams(address = address, size = size))
    })

class P2ESCUWriteDPI(hartId: Int) extends BlackBox with HasBlackBoxInline {
  override def desiredName = s"P2ESCUWriteDPI_$hartId"

  val io = IO(new Bundle {
    val clock      = Input(Clock())
    val reset      = Input(Bool())
    val uart_valid = Input(Bool())
    val uart_data  = Input(UInt(8.W))
    val exit_valid = Input(Bool())
    val exit_code  = Input(UInt(32.W))
  })

  setInline(
    s"P2ESCUWriteDPI_$hartId.v",
    s"""
       |import "DPI-C" context function void p2e_uart_write(input bit [31:0] hart_id, input bit [7:0] ch);
       |import "DPI-C" context function void p2e_sim_exit(input bit [31:0] hart_id, input bit [31:0] code);
       |
       |module P2ESCUWriteDPI_$hartId(
       |  input        clock,
       |  input        reset,
       |  input        uart_valid,
       |  input  [7:0] uart_data,
       |  input        exit_valid,
       |  input [31:0] exit_code
       |);
       |  export "DPI-C" function p2e_scu_${hartId}_hart_id;
       |  function int p2e_scu_${hartId}_hart_id();
       |    p2e_scu_${hartId}_hart_id = $hartId;
       |  endfunction
       |
       |  always @(posedge clock) begin
       |    if (!reset) begin
       |      if (uart_valid) begin
       |        p2e_uart_write($hartId, uart_data);
       |      end
       |
       |      if (exit_valid) begin
       |        p2e_sim_exit($hartId, exit_code);
       |      end
       |    end
       |  end
       |endmodule
    """.stripMargin
  )
}

class TLP2ESCU(params: P2ESCUParams, beatBytes: Int, hartId: Int)(implicit p: Parameters) extends LazyModule {
  require(params.size > 0)
  require(params.size.isValidInt)
  require((params.size & (params.size - 1)) == 0, "P2E SCU size must be a power of two")

  val device = new SimpleDevice(s"p2e-scu-$hartId", Seq("buckyball,p2e-scu"))

  val node = TLRegisterNode(
    address = Seq(AddressSet(params.address, params.size - 1)),
    device = device,
    deviceKey = "reg/control",
    beatBytes = beatBytes
  )

  lazy val module = new LazyModuleImp(this) {
    val uartValid = WireDefault(false.B)
    val uartData  = WireDefault(0.U(8.W))
    val exitValid = WireDefault(false.B)
    val exitCode  = WireDefault(0.U(32.W))

    val dpi = Module(new P2ESCUWriteDPI(hartId))
    dpi.io.clock      := clock
    dpi.io.reset      := reset.asBool
    dpi.io.uart_valid := RegNext(uartValid, false.B)
    dpi.io.uart_data  := RegNext(uartData, 0.U)
    dpi.io.exit_valid := RegNext(exitValid, false.B)
    dpi.io.exit_code  := RegNext(exitCode, 0.U)

    val simExitWrite = RegWriteFn { (valid, data) =>
      exitValid := valid
      exitCode  := data(31, 0)
      true.B
    }

    val uartWrite = RegWriteFn { (valid, data) =>
      uartValid := valid
      uartData  := data(7, 0)
      true.B
    }

    node.regmap(
      0x00000 -> Seq(RegField.w(32, simExitWrite)),
      0x20000 -> Seq(RegField.w(8, uartWrite)),
      0x20005 -> Seq(RegField.r(8, "h60".U))
    )
  }

}
