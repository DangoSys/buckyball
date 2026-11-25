package examples.hart.gemmini

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import examples.hart.gemmini.configs.Params
import framework.hart.hlink.{HLinkIO, HasHLink}
import framework.hart.hlink.configs.{Params => HLinkParam}
import framework.hart.configs.HartParam
import framework.hart.cbus.HartFactory
import buckyball.config.{Design, HartPlacement}
import framework.generic.contract.sims.configs.SimParam

@instantiable
class GemminiHart(p: Params) extends Module with HasHLink {
  @public
  val hlink  = IO(new HLinkIO(HLinkParam(1, false, Seq.empty, p.chi, p.axi, p.axi, false, false)))
  val native = Instantiate(new Native)
  native.io.clock            := clock
  native.io.reset            := reset.asBool
  native.io.io_resetVector   := p.resetVector.U
  native.io.io_interrupts(0) := hlink.interrupts(0).software
  native.io.io_interrupts(1) := hlink.interrupts(0).timer
  native.io.io_interrupts(2) := hlink.interrupts(0).external
  native.io.io_interrupts(3) := hlink.interrupts(0).supervisorExternal
  for (
    (up, down) <- Seq((native.io.memoryAXI(0), hlink.memory.ddr), (native.io.peripheralsAXI(0), hlink.memory.uncached))
  ) {
    down.aw.valid := up.aw.valid
    up.aw.ready   := down.aw.ready
    down.ar.valid := up.ar.valid
    up.ar.ready   := down.ar.ready
    for ((dst, src) <- Seq((down.aw.bits, up.aw.bits), (down.ar.bits, up.ar.bits))) {
      dst       := 0.U.asTypeOf(dst)
      dst.id    := src.id; dst.addr    := src.addr; dst.len   := src.len; dst.size   := src.size
      dst.burst := src.burst; dst.lock := src.lock; dst.cache := src.cache; dst.prot := src.prot; dst.qos := src.qos
    }
    down.w <> up.w
    up.b <> down.b
    up.r <> down.r
  }
}

object GemminiComponent extends HartFactory {
  val component = "gemmini"

  def parameters(design: Design, source: HartPlacement): HartParam = {
    val p = Params.fromProto(source)
    HartParam.fromProto(source, Seq.empty).copy(
      hlink = HLinkParam(1, false, Seq.empty, p.chi, p.axi, p.axi, false, false)
    )
  }

  def instantiate(
    design:   Design,
    source:   HartPlacement,
    hartp:    HartParam,
    simp:     SimParam,
    traceIds: Seq[Int]
  ): HLinkIO =
    Instantiate(new GemminiHart(Params.fromProto(source))).hlink

}
