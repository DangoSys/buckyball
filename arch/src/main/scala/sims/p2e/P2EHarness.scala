package sims.p2e

import chisel3._
import org.chipsalliance.cde.config.Parameters
import chipyard.harness.HasHarnessInstantiators

class P2EHarness(implicit val p: Parameters) extends Module with HasHarnessInstantiators {
  def referenceClockFreqMHz: Double = 1000.0
  def referenceClock:        Clock  = clock
  def referenceReset:        Reset  = reset

  val success  = WireInit(false.B)
  val lazyDuts = instantiateChipTops()
}
