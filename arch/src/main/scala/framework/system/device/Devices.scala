package framework.system.device

import chisel3._
import chisel3.util._
import hier.tile.memory.CoreInterrupts
import memcore.memory.cpu.{CpuMemParams, UncachedRequest, UncachedResponse}

/** Chip devices; the BootROM, when present, is served on the tiles' line paths rather than here. */
case class DeviceParams(
  clint:   ClintParams = ClintParams(),
  plic:    PlicParams = PlicParams(),
  bootrom: Option[BootRomParams] = None)

/**
 * Chip device decode for the per-core uncached device ports. CLINT and PLIC are served inside;
 * every other device address leaves on the same core's external port. Each core port carries one
 * outstanding request, and the internal register blocks take one access per cycle in round robin.
 */
class Devices(p: DeviceParams, cp: CpuMemParams, hartIds: Seq[Int]) extends Module {
  private val n = hartIds.size

  val io = IO(new Bundle {
    val request          = Vec(n, Flipped(Decoupled(new UncachedRequest(cp))))
    val response         = Vec(n, Decoupled(new UncachedResponse(cp)))
    val externalRequest  = Vec(n, Decoupled(new UncachedRequest(cp)))
    val externalResponse = Vec(n, Flipped(Decoupled(new UncachedResponse(cp))))
    val sources          = Input(UInt(p.plic.sources.W))
    val interrupts       = Output(Vec(n, new CoreInterrupts))
  })

  val clint = Module(new Clint(p.clint, hartIds))
  val plic  = Module(new Plic(p.plic, hartIds))
  plic.io.sources := io.sources
  for (i <- 0 until n) {
    io.interrupts(i).timer              := clint.io.mtip(i)
    io.interrupts(i).software           := clint.io.msip(i)
    io.interrupts(i).external           := plic.io.meip(i)
    io.interrupts(i).supervisorExternal := plic.io.seip(i)
  }

  def within(addr: UInt, base: BigInt, bytes: BigInt): Bool = addr >= base.U && addr < (base + bytes).U
  val inClint  = io.request.map(r => within(r.bits.addr, p.clint.base, p.clint.bytes))
  val inPlic   = io.request.map(r => within(r.bits.addr, p.plic.base, p.plic.bytes))
  val internal = inClint.zip(inPlic).map { case (c, l) => c || l }

  val pending = RegInit(VecInit(Seq.fill(n)(false.B)))
  val outside = RegInit(VecInit(Seq.fill(n)(false.B)))
  val tag     = Reg(Vec(n, UInt(cp.tagBits.W)))
  val data    = Reg(Vec(n, UInt(64.W)))
  val error   = Reg(Vec(n, Bool()))
  val idle    = (0 until n).map(i => !pending(i) && !outside(i))

  // Round-robin grant among the cores presenting an internal access this cycle.
  val wants   = VecInit((0 until n).map(i => io.request(i).valid && internal(i) && idle(i))).asUInt
  val last    = RegInit(0.U(log2Ceil(n + 1).W))
  val rotated = (0 until n).map(k => (k.U +& last +& 1.U) % n.U)
  val grant   = PriorityMux(rotated.map(i => wants(i)), rotated)
  val granted = wants.orR
  when(granted)(last := grant)

  val access = io.request(grant).bits
  for (port <- Seq(clint.io.port, plic.io.port)) {
    port.access.bits.addr  := access.addr
    port.access.bits.write := access.write
    port.access.bits.size  := access.size
    port.access.bits.data  := access.data
  }
  // Device registers have no atomics: such an access faults without touching the register.
  val plain = access.atomic === 0.U
  clint.io.port.access.valid := granted && plain && VecInit(inClint)(grant)
  plic.io.port.access.valid  := granted && plain && VecInit(inPlic)(grant)
  val internalRead  = Mux(VecInit(inClint)(grant), clint.io.port.read, plic.io.port.read)
  val internalError = Mux(VecInit(inClint)(grant), clint.io.port.error, plic.io.port.error) || !plain

  for (i <- 0 until n) {
    val request  = io.request(i)
    val response = io.response(i)
    val external = io.externalRequest(i)
    val mine     = granted && grant === i.U
    external.valid               := request.valid && !internal(i) && idle(i)
    external.bits                := request.bits
    request.ready                := idle(i) && Mux(internal(i), mine, external.ready)
    when(request.fire) {
      when(internal(i)) {
        pending(i) := true.B
        tag(i)     := request.bits.tag
        data(i)    := Mux(request.bits.write, 0.U, internalRead)
        error(i)   := internalError
      }.otherwise(outside(i) := true.B)
    }
    response.valid               := pending(i) || (outside(i) && io.externalResponse(i).valid)
    response.bits.tag            := Mux(pending(i), tag(i), io.externalResponse(i).bits.tag)
    response.bits.data           := Mux(pending(i), data(i), io.externalResponse(i).bits.data)
    response.bits.error          := Mux(pending(i), error(i), io.externalResponse(i).bits.error)
    io.externalResponse(i).ready := outside(i) && response.ready
    when(response.fire) { pending(i) := false.B; outside(i) := false.B }
  }
}
