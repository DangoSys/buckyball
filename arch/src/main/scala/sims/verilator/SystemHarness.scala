package sims.verilator

import chisel3._
import chisel3.experimental.hierarchy.Instantiate
import sims.soc.{SimSoc, SystemTarget}

/** Verilator top for the explicit System. Keeps the BBSimHarness name bebop builds against. */
class SystemHarness(target: SystemTarget, diffTest: Boolean) extends Module {
  override def desiredName = "BBSimHarness"

  val bdbClkDpi = Instantiate(new BdbClkDPI)
  bdbClkDpi.io.clock := clock
  bdbClkDpi.io.reset := reset.asBool

  val soc = Module(new SimSoc(target, diffTest))

  val dram = Instantiate(new BBSimDRAM(target.dramBytes, 64, 1000, target.dramBase, soc.axiParams, chipId = 0))
  dram.io.clock := clock
  dram.io.reset := reset
  dram.io.axi <> soc.io.axi
}
