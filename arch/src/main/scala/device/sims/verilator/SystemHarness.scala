package framework.device.sims.verilator

import framework.device.sims.configs.GlobalConfig

import chisel3._
import chisel3.experimental.hierarchy.Instantiate
import chisel3.experimental.hierarchy.Instance
import framework.device.sims.soc.{SimSoc, SystemTarget}

/** Verilator top for the explicit System. Keeps the BBSimHarness name bebop builds against. */
class SystemHarness(b: GlobalConfig, target: SystemTarget) extends Module {
  val diffTest = b.sim.diffTest

  override def desiredName = "BBSimHarness"

  val bdbClkDpi: Instance[BdbClkDPI] = Instantiate(new BdbClkDPI)
  bdbClkDpi.io.clock := clock
  bdbClkDpi.io.reset := reset.asBool

  // The system's reset comes from a register, not from the model's reset input. Verilator evaluates what depends on a
  // model input in one serial pass at every evaluation, and a hierarchical block takes its reset as an input: wired
  // from the pin, every block of the design would be evaluated in that pass, one after another.
  val systemReset = RegNext(reset.asBool, true.B)

  val soc = withReset(systemReset) {
    Instantiate(new SimSoc(
      b,
      target
    ))
  }

  val dram: Instance[BBSimDRAM] =
    Instantiate(new BBSimDRAM(b.device.ddr.port, b.device.ddr.base, b.device.ddr.bytes, b.sim.memory, 0))

  dram.io.clock := clock
  dram.io.reset := reset
  dram.io.axi <> soc.io.axi
}
