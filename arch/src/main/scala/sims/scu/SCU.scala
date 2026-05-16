package sims.scu

import chisel3._
import chisel3.util.{log2Ceil, HasBlackBoxInline, Mux1H, PriorityEncoder}
import org.chipsalliance.cde.config.{Config, Field, Parameters}

import freechips.rocketchip.diplomacy._
import freechips.rocketchip.regmapper._
import freechips.rocketchip.subsystem._
import freechips.rocketchip.tilelink._

/**
 * SCU is a global multi-hart System Control Unit. A single instance attaches to
 * a system bus (typically CBUS) and provides a per-hart sub-region for UART
 * output and simulation exit. Each hart accesses its own sub-region via
 * `baseAddress + hartId * strideBytes`.
 *
 * Address layout (default values):
 *   baseAddress     = 0x60000000
 *   strideBytes     = 0x40000   (256 KiB per hart)
 *   totalSizeBytes  = 0x10000000 (256 MiB total, supports up to 1024 harts)
 *   maxHarts        = 64
 *
 * Within each hart's sub-region:
 *   +0x00000: simExit register (write triggers $finish with exit code)
 *   +0x20000: uartTx register  (write outputs a byte via DPI-C)
 *   +0x20005: status register  (read-only, returns 0x60)
 *
 * @param baseAddress   physical base address (must be aligned to totalSizeBytes)
 * @param strideBytes   bytes per hart (must be power of two)
 * @param totalSizeBytes total address window (must be power of two,
 *                       >= maxHarts * strideBytes)
 * @param maxHarts      number of DPI instances to elaborate. Accesses targeting
 *                      hartId >= maxHarts return a TileLink access-denied error
 *                      via the address decoder (those addresses are simply not
 *                      claimed by this manager).
 */
case class SCUParams(
  baseAddress:    BigInt = BigInt("60000000", 16),
  strideBytes:    BigInt = BigInt("40000", 16),
  totalSizeBytes: BigInt = BigInt("10000000", 16),
  maxHarts:       Int = 64) {
  require(
    strideBytes > 0 && (strideBytes & (strideBytes - 1)) == 0,
    s"SCU strideBytes ($strideBytes) must be a positive power of two"
  )
  require(
    totalSizeBytes > 0 && (totalSizeBytes & (totalSizeBytes - 1)) == 0,
    s"SCU totalSizeBytes ($totalSizeBytes) must be a positive power of two"
  )
  require(maxHarts > 0, s"SCU maxHarts ($maxHarts) must be positive")
  require(
    BigInt(maxHarts) * strideBytes <= totalSizeBytes,
    s"SCU maxHarts ($maxHarts) * strideBytes ($strideBytes) = " +
      s"${BigInt(maxHarts) * strideBytes} exceeds totalSizeBytes ($totalSizeBytes)"
  )
}

case object SCUKey extends Field[Option[SCUParams]](None)

/**
 * Config fragment that sets SCU parameters AND replaces the default DigitalTop
 * with one that includes the SCU on CBUS.
 */
class WithSCU(
  baseAddress:    BigInt = BigInt("60000000", 16),
  strideBytes:    BigInt = BigInt("40000", 16),
  totalSizeBytes: BigInt = BigInt("10000000", 16),
  maxHarts:       Int = 64)
    extends Config((site, here, up) => {
      case SCUKey               => Some(SCUParams(
          baseAddress = baseAddress,
          strideBytes = strideBytes,
          totalSizeBytes = totalSizeBytes,
          maxHarts = maxHarts
        ))
      case chipyard.BuildSystem => (p: Parameters) => new DigitalTopWithSCU()(p)
    })

/**
 * DigitalTop subclass that mixes in CanHavePeripherySCU. This avoids modifying
 * any chipyard source files.
 */
class DigitalTopWithSCU(implicit p: Parameters) extends chipyard.DigitalTop with CanHavePeripherySCU

/**
 * Single DPI-C bridge module for all harts. The hart_id is supplied as an
 * input signal rather than being baked into the module name, so only one
 * BlackBox is generated and DPI-C imports are not duplicated.
 *
 * Has separate hart_id inputs for uart and exit operations because different
 * harts may write uart and exit simultaneously.
 */
class SCUWriteDPI extends BlackBox with HasBlackBoxInline {
  override def desiredName = "SCUWriteDPI"

  val io = IO(new Bundle {
    val clock        = Input(Clock())
    val reset        = Input(Bool())
    val uart_hart_id = Input(UInt(32.W))
    val uart_valid   = Input(Bool())
    val uart_data    = Input(UInt(8.W))
    val exit_hart_id = Input(UInt(32.W))
    val exit_valid   = Input(Bool())
    val exit_code    = Input(UInt(32.W))
  })

  setInline(
    "SCUWriteDPI.v",
    s"""
       |import "DPI-C" context function void scu_uart_write(input int unsigned hart_id, input int unsigned ch);
       |import "DPI-C" context function void scu_sim_exit(input int unsigned hart_id, input int unsigned code);
       |
       |module SCUWriteDPI(
       |  input         clock,
       |  input         reset,
       |  input  [31:0] uart_hart_id,
       |  input         uart_valid,
       |  input  [7:0]  uart_data,
       |  input  [31:0] exit_hart_id,
       |  input         exit_valid,
       |  input  [31:0] exit_code
       |);
       |  always @(posedge clock) begin
       |    if (!reset) begin
       |      if (uart_valid) begin
       |        scu_uart_write(uart_hart_id, {24'h0, uart_data});
       |      end
       |
       |      if (exit_valid) begin
       |        scu_sim_exit(exit_hart_id, exit_code);
       |      end
       |    end
       |  end
       |endmodule
    """.stripMargin
  )
}

/**
 * Global multi-hart SCU. Each hart owns a sub-region of size `strideBytes`
 * starting at `baseAddress + hartId * strideBytes`. Address decoder only
 * claims addresses for `hartId < maxHarts`; accesses to higher hart IDs (still
 * within `totalSizeBytes`) fall through and are answered by the bus's
 * unmapped-address error device.
 */
class TLSCU(params: SCUParams, beatBytes: Int)(implicit p: Parameters) extends LazyModule {
  val device = new SimpleDevice("scu", Seq("buckyball,scu"))

  // One AddressSet per hart, exactly covering that hart's stride region.
  // Diplomacy will route only addresses within these sets to this manager;
  // anything else in the totalSizeBytes window is left unmapped.
  val perHartAddresses: Seq[AddressSet] = (0 until params.maxHarts).map { h =>
    AddressSet(params.baseAddress + BigInt(h) * params.strideBytes, params.strideBytes - 1)
  }

  val node = TLRegisterNode(
    address = perHartAddresses,
    device = device,
    deviceKey = "reg/control",
    beatBytes = beatBytes,
    concurrency = 1
  )

  lazy val module = new LazyModuleImp(this) {
    // Single DPI instance shared by all harts. We multiplex per-hart signals
    // using priority encoders (lower hart ID wins if multiple harts write
    // simultaneously, though this should be rare in practice).
    val dpi = Module(new SCUWriteDPI)
    dpi.io.clock := clock
    dpi.io.reset := reset.asBool

    // Collect per-hart valid/data signals
    val uartValids = Wire(Vec(params.maxHarts, Bool()))
    val uartDatas  = Wire(Vec(params.maxHarts, UInt(8.W)))
    val exitValids = Wire(Vec(params.maxHarts, Bool()))
    val exitCodes  = Wire(Vec(params.maxHarts, UInt(32.W)))

    // Default: no hart is active
    uartValids.foreach(_ := false.B)
    uartDatas.foreach(_  := 0.U)
    exitValids.foreach(_ := false.B)
    exitCodes.foreach(_  := 0.U)

    // Mux per-hart signals into the single DPI
    dpi.io.uart_valid   := RegNext(uartValids.asUInt.orR, false.B)
    dpi.io.uart_hart_id := RegNext(PriorityEncoder(uartValids.asUInt), 0.U)
    dpi.io.uart_data    := RegNext(Mux1H(uartValids, uartDatas), 0.U)
    dpi.io.exit_valid   := RegNext(exitValids.asUInt.orR, false.B)
    dpi.io.exit_hart_id := RegNext(PriorityEncoder(exitValids.asUInt), 0.U)
    dpi.io.exit_code    := RegNext(Mux1H(exitValids, exitCodes), 0.U)

    // Per-hart registers and write functions
    val allFields: Seq[RegField.Map] =
      (0 until params.maxHarts).flatMap { h =>
        val hartBase = (params.baseAddress + BigInt(h) * params.strideBytes).toInt

        val simExitReg   = RegInit(0.U(32.W))
        val simExitWrite = RegWriteFn { (valid, data) =>
          exitValids(h) := valid
          exitCodes(h)  := data(31, 0)
          simExitReg    := data(31, 0)
          true.B
        }

        val uartTxReg = RegInit(0.U(8.W))
        val uartWrite = RegWriteFn { (valid, data) =>
          uartValids(h) := valid
          uartDatas(h)  := data(7, 0)
          uartTxReg     := data(7, 0)
          true.B
        }

        // Each hart's registers at hartBase + offset
        Seq(
          (hartBase + 0x00000) -> Seq(RegField(32, simExitReg, simExitWrite)),
          (hartBase + 0x20000) -> Seq(RegField(8, uartTxReg, uartWrite)),
          (hartBase + 0x20005) -> Seq(RegField.r(8, "h60".U))
        )
      }

    // Register all fields at once
    node.regmap(allFields: _*)
  }

}

/**
 * Attach a single global SCU instance to CBUS.
 * This trait must be mixed into the subsystem at construction time. We do this
 * by defining a custom subsystem that extends ChipyardSubsystem with this trait,
 * and then a config fragment that selects this subsystem via BuildSystem.
 *
 * Pattern follows CanHavePeripheryCLINT: create a synchronous domain wrapper
 * tied to the target bus's clock, then instantiate the SCU inside it so the
 * implicit clock/reset are provided correctly.
 */
trait CanHavePeripherySCU { this: BaseSubsystem =>

  val scuOpt = p(SCUKey).map { params =>
    val tlbus            = locateTLBusWrapper(CBUS)
    val scuDomainWrapper = tlbus.generateSynchronousDomain("SCU").suggestName("scu_domain")
    val scu              = scuDomainWrapper(LazyModule(new TLSCU(params, tlbus.beatBytes)))
    scuDomainWrapper {
      scu.node := tlbus.coupleTo("scu")(TLFragmenter(tlbus, Some("SCU")) := _)
    }
    scu
  }

}
