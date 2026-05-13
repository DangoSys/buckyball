package sims.verilator

import chisel3._

import org.chipsalliance.cde.config.{Config, Parameters}

import chipyard.harness.{HarnessBinder, HasHarnessInstantiators}
import chipyard.iobinders.AXI4MemPort
import sims.scu.WithSCU

class WithBBSimMem
    extends HarnessBinder({
      case (th: HasHarnessInstantiators, port: AXI4MemPort, chipId: Int) => {
        val memSize   = port.params.master.size
        val memBase   = port.params.master.base
        val lineSize  = 64
        val clockFreq = port.clockFreqMHz
        val mem       = Module(
          new BBSimDRAM(memSize, lineSize, clockFreq, memBase, port.edge.bundle, chipId)
        ).suggestName("bbsimdram")

        mem.io.clock := port.io.clock
        mem.io.reset := th.harnessBinderReset.asAsyncReset
        mem.io.axi <> port.io.bits
      }
    })

// =============================================================================
// BBSimConfig: Verilator harness-level config.
// Concrete VerilatorConfigs (e.g. BuckyballToyVerilatorConfig) compose this
// with the SoC config (which in turn includes BuckyballBaseConfig).
//
// Verilator now uses the P2E SCU (per-tile UART/exit via DPI-C) instead of
// the AXI4 MMIO port. The SCU is intercepted inside each BBTile and never
// reaches the system bus.
// =============================================================================
class BBSimConfig
    extends Config(
      new WithSCU ++
        new WithBBSimMem ++
        new chipyard.config.WithUniformBusFrequencies(1000.0) ++
        new chipyard.harness.WithTieOffInterrupts ++
        new chipyard.harness.WithTieOffL2FBusAXI ++
        new chipyard.harness.WithClockFromHarness ++
        new chipyard.harness.WithResetFromHarness ++
        new chipyard.harness.WithAbsoluteFreqHarnessClockInstantiator ++
        new chipyard.iobinders.WithAXI4MemPunchthrough ++
        new chipyard.iobinders.WithNMITiedOff
    )

// =============================================================================
// BBSimHarness
// =============================================================================
class BBSimHarness(implicit val p: Parameters) extends Module with HasHarnessInstantiators {

  val bdbClkDpi = Module(new BdbClkDPI)
  bdbClkDpi.io.clock := clock
  bdbClkDpi.io.reset := reset.asBool

  def referenceClockFreqMHz: Double = 1000.0
  def referenceClock:        Clock  = clock
  def referenceReset:        Reset  = reset

  val success = WireInit(false.B)

  val lazyDuts = instantiateChipTops()
}
