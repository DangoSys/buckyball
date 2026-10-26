package framework.acc.memdomain.tol1

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.ip.memory.interlock.{
  AccessInfo,
  Acknowledgement,
  CpuQuery,
  Dispatch,
  Interlock,
  Maintenance,
  Params,
  Tag
}

class Port(val p: Params) extends Bundle {

  val dispatch      = Flipped(Decoupled(new Dispatch(p)))
  val cancel        = Flipped(Decoupled(new Dispatch(p)))
  val accessInfo    = Flipped(Decoupled(new AccessInfo(p)))
  val cpuQuery      = Input(new CpuQuery(p))
  val cpuAllow      = Output(Bool())
  val cpuProbeAllow = Output(Bool())
  val maintenance   = Decoupled(new Maintenance(p))
  val maintained    =
    Flipped(Decoupled(new Acknowledgement(p)))
  val grant         = Decoupled(new Tag(p))
  val done          = Flipped(Decoupled(new Acknowledgement(p)))
  val complete      = Decoupled(new Tag(p))
}

@instantiable
class ToL1(val p: Params) extends Module {

  @public
  val io = IO(new Port(p))

  val interlock: Instance[Interlock] = Instantiate(new Interlock(p))

  interlock.io.dispatch <> io.dispatch
  interlock.io.cancel <> io.cancel
  interlock.io.accessInfo <> io.accessInfo
  interlock.io.cpuQuery := io.cpuQuery
  io.cpuAllow           := interlock.io.cpuAllow
  io.cpuProbeAllow      := interlock.io.cpuProbeAllow
  io.maintenance <> interlock.io.maintenance
  interlock.io.maintained <> io.maintained
  io.grant <> interlock.io.grant
  interlock.io.done <> io.done
  io.complete <> interlock.io.complete
}
