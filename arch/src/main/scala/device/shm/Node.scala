package framework.device.shm

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import framework.generic.bus.axi4

/** A device communication endpoint with its own storage and copy kernel. */
@instantiable
class Node(p: Params) extends Module {

  @public val io = IO(new Bundle {
    val request = Flipped(Decoupled(new ControlRequest))
    val reply   = Decoupled(UInt(64.W))
    val busy    = Output(Bool())
    val local   = Flipped(new StorageAccess(p.axi.dataBits))
    val hartId  = Input(UInt(p.hartBits.W))
    val link    = new Link(p.axi)
  })

  val memory  = Instantiate(new Storage(p, 5))
  val control = Instantiate(new Control(p))
  val copy    = Instantiate(new C2CTransfer(p))
  val receive = Instantiate(new SharedAxiEndpoint(p))
  control.io.request <> io.request
  io.reply <> control.io.reply
  control.io.hartId         := io.hartId
  io.link.routeTarget       := control.io.routeTarget
  control.io.routeAvailable := io.link.routeAvailable
  copy.io.command <> control.io.command
  control.io.completion <> copy.io.completion
  memory.io.access(0) <> copy.io.source
  memory.io.access(1) <> receive.io.read
  memory.io.access(2) <> receive.io.write
  memory.io.access(3) <> control.io.local
  memory.io.access(4) <> io.local
  io.link.tx <> copy.io.mem
  receive.io.mem <> io.link.rx
  io.busy                   := !control.io.idle || receive.io.busy || memory.io.busy
}
