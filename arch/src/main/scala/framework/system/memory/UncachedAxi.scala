package framework.system.memory

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import memcore.bus.axi4
import memcore.memory.cpu.{CpuMemParams, UncachedRequest, UncachedResponse}

/** Uncached CPU accesses share one AXI master; responses retain the CPU port and tag. */
@instantiable
class UncachedAxi(cp: CpuMemParams, ap: axi4.Params, ports: Int) extends Module {

  @public
  val io = IO(new Bundle {
    val request  = Vec(ports, Flipped(Decoupled(new UncachedRequest(cp))))
    val response = Vec(ports, Decoupled(new UncachedResponse(cp)))
    val axi      = new axi4.Port(ap)
  })

  val indexBits                                       = math.max(1, log2Ceil(ports))
  val idle :: issue :: waitResponse :: respond :: Nil = Enum(4)
  val state                                           = RegInit(idle)
  val cursor                                          = RegInit(0.U(indexBits.W))
  val owner                                           = Reg(UInt(indexBits.W))
  val command                                         = Reg(new UncachedRequest(cp))
  val answer                                          = Reg(new UncachedResponse(cp))
  val addressSent                                     = RegInit(false.B)
  val dataSent                                        = RegInit(false.B)
  val ordered                                         = (0 until ports).map(i => (cursor +& i.U) % ports.U)
  val selected                                        = if (ports == 1) 0.U else PriorityMux(ordered.map(i => io.request(i).valid), ordered)
  val present                                         = io.request.map(_.valid).reduce(_ || _)
  for (i       <- 0 until ports) {
    io.request(i).ready             := state === idle && present && selected === i.U
    io.response(i).valid            := state === respond && owner === i.U
    io.response(i).bits             := answer
    when(io.response(i).fire)(state := idle)
  }
  when(state === idle && present) {
    owner       := selected
    command     := (if (ports == 1) io.request.head.bits else io.request(selected).bits)
    addressSent := false.B
    dataSent    := false.B
    cursor      := Mux(selected === (ports - 1).U, 0.U, selected + 1.U)
    state       := issue
  }
  for (address <- Seq(io.axi.aw, io.axi.ar)) {
    address.bits       := 0.U.asTypeOf(address.bits)
    address.bits.addr  := command.addr
    address.bits.size  := command.size
    address.bits.burst := 1.U
  }
  io.axi.aw.valid := state === issue && command.write && !addressSent
  io.axi.ar.valid := state === issue && !command.write
  val shift = command.addr(log2Ceil(ap.bytes) - 1, 0)
  io.axi.w.valid                   := state === issue && command.write && !dataSent
  io.axi.w.bits.data               := command.data << (shift << 3)
  io.axi.w.bits.strb               := (((1.U((ap.bytes + 1).W) << (1.U << command.size)) - 1.U) << shift)
  io.axi.w.bits.last               := true.B
  when(io.axi.aw.fire)(addressSent := true.B)
  when(io.axi.w.fire)(dataSent     := true.B)
  when(io.axi.ar.fire || (command.write && (addressSent || io.axi.aw.fire) && (dataSent || io.axi.w.fire))) {
    when(state === issue)(state := waitResponse)
  }
  io.axi.r.ready                   := state === waitResponse && !command.write
  io.axi.b.ready                   := state === waitResponse && command.write
  when(io.axi.r.fire || io.axi.b.fire) {
    when(io.axi.r.fire)(assert(io.axi.r.bits.last))
    answer.tag   := command.tag
    answer.data  := Mux(command.write, 0.U, io.axi.r.bits.data >> (shift << 3))
    answer.error := Mux(command.write, io.axi.b.bits.resp =/= 0.U, io.axi.r.bits.resp =/= 0.U)
    state        := respond
  }
}
