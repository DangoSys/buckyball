package sims.scu

import chisel3._
import chisel3.util.HasBlackBoxInline
import org.chipsalliance.cde.config.{Config, Field, Parameters}

import freechips.rocketchip.devices.tilelink._
import freechips.rocketchip.diplomacy._
import freechips.rocketchip.regmapper._
import freechips.rocketchip.subsystem._
import freechips.rocketchip.tilelink._

case class SCUParams(
  address: BigInt = BigInt("60000000", 16),
  size:    BigInt = BigInt("40000", 16))

case object SCUKey extends Field[Option[SCUParams]](None)

class WithSCU(
  address: BigInt = BigInt("60000000", 16),
  size:    BigInt = BigInt("40000", 16))
    extends Config((site, here, up) => {
      case SCUKey => Some(SCUParams(address = address, size = size))
    })

class SCUWriteDPI(hartId: Int) extends BlackBox with HasBlackBoxInline {
  override def desiredName = s"SCUWriteDPI_$hartId"

  val io = IO(new Bundle {
    val clock      = Input(Clock())
    val reset      = Input(Bool())
    val uart_valid = Input(Bool())
    val uart_data  = Input(UInt(8.W))
    val exit_valid = Input(Bool())
    val exit_code  = Input(UInt(32.W))
  })

  setInline(
    s"SCUWriteDPI_$hartId.v",
    s"""
       |import "DPI-C" context function void scu_uart_write(input int unsigned hart_id, input int unsigned ch);
       |import "DPI-C" context function void scu_sim_exit(input int unsigned hart_id, input int unsigned code);
       |
       |module SCUWriteDPI_$hartId(
       |  input        clock,
       |  input        reset,
       |  input        uart_valid,
       |  input  [7:0] uart_data,
       |  input        exit_valid,
       |  input [31:0] exit_code
       |);
       |  export "DPI-C" function scu_${hartId}_hart_id;
       |  function int scu_${hartId}_hart_id();
       |    scu_${hartId}_hart_id = $hartId;
       |  endfunction
       |
       |  always @(posedge clock) begin
       |    if (!reset) begin
       |      if (uart_valid) begin
       |        scu_uart_write($hartId, {24'h0, uart_data});
       |      end
       |
       |      if (exit_valid) begin
       |        scu_sim_exit($hartId, exit_code);
       |      end
       |    end
       |  end
       |endmodule
    """.stripMargin
  )
}

class TLSCU(params: SCUParams, beatBytes: Int, hartId: Int)(implicit p: Parameters) extends LazyModule {
  require(params.size > 0)
  require(params.size.isValidInt)
  require((params.size & (params.size - 1)) == 0, "SCU size must be a power of two")

  val device = new SimpleDevice(s"scu-$hartId", Seq("buckyball,scu"))

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

    val dpi = Module(new SCUWriteDPI(hartId))
    dpi.io.clock      := clock
    dpi.io.reset      := reset.asBool
    dpi.io.uart_valid := RegNext(uartValid, false.B)
    dpi.io.uart_data  := RegNext(uartData, 0.U)
    dpi.io.exit_valid := RegNext(exitValid, false.B)
    dpi.io.exit_code  := RegNext(exitCode, 0.U)

    val simExitReg = RegInit(0.U(32.W))

    val simExitWrite = RegWriteFn { (valid, data) =>
      exitValid  := valid
      exitCode   := data(31, 0)
      simExitReg := data(31, 0)
      true.B
    }

    val uartTxReg = RegInit(0.U(8.W))

    val uartWrite = RegWriteFn { (valid, data) =>
      uartValid := valid
      uartData  := data(7, 0)
      uartTxReg := data(7, 0)
      true.B
    }

    node.regmap(
      0x00000 -> Seq(RegField(32, simExitReg, simExitWrite)), // read/write
      0x20000 -> Seq(RegField(8, uartTxReg, uartWrite)),      // read/write
      0x20005 -> Seq(RegField.r(8, "h60".U))                  // read-only status
    )
  }

}
