package sims.verilator

import chisel3._
// _root_ disambiguates from package chisel3.util.circt if user imports chisel3.util._
import _root_.circt.stage.ChiselStage
import org.chipsalliance.cde.config.Parameters

object Elaborate extends App {
  val config = new examples.toy.BuckyBallToyConfig
  val params = config.toInstance
  
  ChiselStage.emitSystemVerilogFile(
    new chipyard.harness.TestHarness()(config.toInstance),
    firtoolOpts = args,
    args = Array.empty  // directly pass command line arguments to firtool
  )
}       
