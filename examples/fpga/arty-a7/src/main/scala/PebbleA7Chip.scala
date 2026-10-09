package fpga.arty

import chisel3._
import circt.stage.ChiselStage
import org.chipsalliance.cde.config.Parameters
import org.chipsalliance.diplomacy.lazymodule.LazyModule
import freechips.rocketchip.system.ExampleRocketSystem

// Real CPU + TileLink + Buckyball accelerator. The two AXI masters are
// terminated by synthesizable board memory/peripherals, never SimAXIMem.
class PebbleA7Chip(implicit p: Parameters) extends Module {
  val system      = LazyModule(new ExampleRocketSystem)
  val core        = Module(system.module)
  system.io_clocks.get.elements.values.foreach { domain =>
    domain.clock := clock
    domain.reset := reset.asBool
  }
  core.tieOffInterrupts()
  require(system.mem_axi4.size == 1, "Arty requires one AXI memory port")
  require(system.mmio_axi4.size == 1, "Arty requires one AXI peripheral port")
  val memory      = IO(chiselTypeOf(system.mem_axi4.head))
  val peripherals = IO(chiselTypeOf(system.mmio_axi4.head))
  memory <> system.mem_axi4.head
  peripherals <> system.mmio_axi4.head
}

object EmitPebbleA7 extends App {
  implicit val parameters: Parameters = new PebbleA7Config().toInstance
  ChiselStage.emitSystemVerilogFile(
    new PebbleA7Chip,
    args = Array("--target-dir", args.headOption.getOrElse("generated")),
    firtoolOpts = Array("-disable-all-randomization", "-strip-debug-info")
  )
}
