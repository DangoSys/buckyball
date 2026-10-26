package framework.device.peripherals

import chisel3.experimental.hierarchy.Instance

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import framework.device.peripherals.clint.Clint
import framework.device.peripherals.plic.Plic
import framework.device.peripherals.scu.SystemControl
import framework.ip.bus.axi4
import framework.device.configs.DeviceParam
import framework.device.peripherals.plic.configs.PlicParams
import framework.core.clink.CoreInterrupts
import framework.core.memory.access.{AccessParams, UncachedRequest, UncachedResponse}

/** One register access presented for a single cycle; data is right-aligned at `addr`. */
class DeviceAccess extends Bundle {
  val addr  = UInt(64.W)
  val write = Bool()
  val size  = UInt(3.W)
  val data  = UInt(64.W)
}

/** Register-block interface: `read` and `error` answer the access presented in the same cycle. */
class DevicePort extends Bundle {
  val access = Flipped(Valid(new DeviceAccess))
  val read   = Output(UInt(64.W))
  val error  = Output(Bool())
}

/** Device interrupt events and the resulting per-hart interrupt notifications. */
class HardwareInterrupt(p: PlicParams) extends Bundle {
  val sourceCount = p.sources
  val hartCount   = p.hartIds.size
  val sources     = Input(UInt(sourceCount.W))
  val harts       = Output(Vec(hartCount, new CoreInterrupts))
}

/** AXI4 register endpoint: single transfers up to 64 bits, with full transfer strobes or a zero-strobe no-op. */
@instantiable
class Controller(device: DeviceParam, ap: axi4.Params, cp: AccessParams) extends Module {
  val hartIds = device.plic.hartIds

  @public
  val io = IO(new Bundle {
    val axi               = Flipped(new axi4.Port(ap))
    val hardwareInterrupt = new HardwareInterrupt(device.plic)
  })

  require(ap.dataBits >= 64 && cp.tagBits >= ap.idBits)
  val request  = Wire(Decoupled(new UncachedRequest(cp)))
  val response = Wire(Decoupled(new UncachedResponse(cp)))
  val aw       = Reg(new axi4.Address(ap))
  val ar       = Reg(new axi4.Address(ap))
  val wd       = Reg(new axi4.WriteData(ap))
  val awHeld   = RegInit(false.B)
  val arHeld   = RegInit(false.B)
  val wHeld    = RegInit(false.B)
  io.axi.aw.ready := !awHeld
  io.axi.ar.ready := !arHeld
  io.axi.w.ready  := !wHeld
  when(io.axi.aw.fire) {
    assert(io.axi.aw.bits.len === 0.U && !io.axi.aw.bits.lock, "Peripheral AXI accepts single nonexclusive transfers")
    aw := io.axi.aw.bits; awHeld := true.B
  }
  when(io.axi.ar.fire) {
    assert(io.axi.ar.bits.len === 0.U && !io.axi.ar.bits.lock, "Peripheral AXI accepts single nonexclusive transfers")
    ar := io.axi.ar.bits; arHeld := true.B
  }
  when(io.axi.w.fire) {
    assert(io.axi.w.bits.last, "Peripheral AXI write must contain one beat")
    wd := io.axi.w.bits; wHeld := true.B
  }

  val bPending = RegInit(false.B)
  val rPending = RegInit(false.B)
  val bAnswer  = Reg(new axi4.WriteResponse(ap))
  val rAnswer  = Reg(new axi4.ReadData(ap))
  io.axi.b.valid               := bPending
  io.axi.b.bits                := bAnswer
  io.axi.r.valid               := rPending
  io.axi.r.bits                := rAnswer
  when(io.axi.b.fire)(bPending := false.B)
  when(io.axi.r.fire)(rPending := false.B)

  val active        = RegInit(false.B)
  val preferWrite   = RegInit(false.B)
  val canWrite      = awHeld && wHeld && !bPending
  val canRead       = arHeld && !rPending
  val writing       = canWrite && (!canRead || preferWrite)
  val address       = Mux(writing, aw, ar)
  val lane          = address.addr(log2Ceil(ap.bytes) - 1, 0)
  val expectedMask  = (((1.U((ap.bytes + 1).W) << (1.U << address.size)) - 1.U) << lane)(ap.bytes - 1, 0)
  val noWrite       = writing && wd.strb === 0.U
  val invalidAccess = address.size > 3.U || (address.addr & ((1.U(ap.addressBits.W) << address.size) - 1.U)).orR ||
    (writing && !noWrite && wd.strb =/= expectedMask)
  request.valid      := !active && (canWrite || canRead)
  request.bits.addr  := address.addr
  request.bits.tag   := address.id
  request.bits.size  := address.size
  request.bits.write := writing
  request.bits.data  := (wd.data >> (lane << 3))(63, 0)
  val responseWrite = Reg(Bool())
  val responseId    = Reg(UInt(ap.idBits.W))
  val responseLane  = Reg(UInt(log2Ceil(ap.bytes).W))
  when(request.fire) {
    active        := true.B
    preferWrite   := !writing
    responseWrite := writing
    responseId    := address.id
    responseLane  := lane
    when(writing) { awHeld := false.B; wHeld := false.B }.otherwise(arHeld := false.B)
  }
  response.ready := active && Mux(responseWrite, !bPending, !rPending)
  when(response.fire) {
    active := false.B
    when(responseWrite) {
      bPending     := true.B
      bAnswer.id   := responseId
      bAnswer.resp := Mux(response.bits.error, 2.U, 0.U)
    }.otherwise {
      rPending     := true.B
      rAnswer.id   := responseId
      rAnswer.data := response.bits.data << (responseLane << 3)
      rAnswer.resp := Mux(response.bits.error, 2.U, 0.U)
      rAnswer.last := true.B
    }
  }

  val clint: Instance[Clint]         = Instantiate(new Clint(device.clint))
  val plic:  Instance[Plic]          = Instantiate(new Plic(device.plic))
  val scu:   Instance[SystemControl] = Instantiate(new SystemControl(device.scu, cp))
  plic.io.sources := io.hardwareInterrupt.sources
  for (i <- hartIds.indices) {
    io.hardwareInterrupt.harts(i).timer              := clint.io.mtip(i)
    io.hardwareInterrupt.harts(i).software           := clint.io.msip(i)
    io.hardwareInterrupt.harts(i).external           := plic.io.meip(i)
    io.hardwareInterrupt.harts(i).supervisorExternal := plic.io.seip(i)
  }

  val addr        = request.bits.addr
  val inClint     = addr >= device.clint.base.U && addr < (device.clint.base + device.clint.bytes).U
  val inPlic      = addr >= device.plic.base.U && addr < (device.plic.base + device.plic.bytes).U
  val inScu       = addr >= device.scu.baseAddress.U && addr < (device.scu.baseAddress + device.scu.totalSizeBytes).U
  val pending     = RegInit(false.B)
  val selectedScu = Reg(Bool())
  val answer      = Reg(new UncachedResponse(cp))

  request.ready        := !pending && Mux(inScu && !invalidAccess && !noWrite, scu.io.request.ready, true.B)
  scu.io.request.valid := request.valid && !pending && inScu && !invalidAccess && !noWrite
  scu.io.request.bits  := request.bits
  for (port <- Seq(clint.io.port, plic.io.port)) {
    port.access.bits.addr  := addr
    port.access.bits.write := request.bits.write
    port.access.bits.size  := request.bits.size
    port.access.bits.data  := request.bits.data
  }
  clint.io.port.access.valid := request.fire && inClint && !invalidAccess && !noWrite
  plic.io.port.access.valid   := request.fire && inPlic && !invalidAccess && !noWrite
  when(request.fire) {
    pending      := true.B
    selectedScu  := inScu && !invalidAccess && !noWrite
    answer.tag   := request.bits.tag
    answer.data  := Mux(request.bits.write, 0.U, Mux(inClint, clint.io.port.read, plic.io.port.read))
    answer.error := invalidAccess || (!noWrite && Mux(
      inClint,
      clint.io.port.error,
      Mux(inPlic, plic.io.port.error, !inScu)
    ))
  }
  response.valid              := pending && Mux(selectedScu, scu.io.response.valid, true.B)
  response.bits               := Mux(selectedScu, scu.io.response.bits, answer)
  scu.io.response.ready       := pending && selectedScu && response.ready
  when(response.fire)(pending := false.B)
}
