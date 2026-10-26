package framework.ip.bus.chi.rnf

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import framework.ip.bus.axi4
import framework.ip.bus.chi.RequesterPort

/**
 * One independent AXI read and write burst, served through a coherent cache.
 * AXI beats are ordinary memory accesses; 128-bit writes are not atomic.
 */
@instantiable
class AxiCache(p: axi4.Params, cacheParam: RnfParams) extends Module {
  require(p.dataBits == 128 && p.addressBits == cacheParam.chi.addressBits)
  require(cacheParam.lineResult && !cacheParam.probe)

  @public
  val io = IO(new Bundle {
    val axi = Flipped(new axi4.Port(p))
    val chi = new RequesterPort(cacheParam.chi)
  })

  val cache = Instantiate(new BankedChiCache(cacheParam))
  io.chi <> cache.io.chi

  val reading   = RegInit(false.B)
  val read      = Reg(new axi4.Address(p))
  val readBeat  = RegInit(0.U(8.W))
  val readValid = RegInit(false.B)
  val readData  = Reg(UInt(128.W))
  val readError = Reg(Bool())
  io.axi.ar.ready    := !reading
  when(io.axi.ar.fire) {
    assert(io.axi.ar.bits.size <= 4.U && (io.axi.ar.bits.burst === 1.U || io.axi.ar.bits.len === 0.U))
    assert((io.axi.ar.bits.addr & ((1.U(17.W) << io.axi.ar.bits.size) - 1.U)) === 0.U)
    assert(!io.axi.ar.bits.lock && io.axi.ar.bits.region === 0.U && io.axi.ar.bits.cache === 15.U)
    read     := io.axi.ar.bits
    reading  := true.B
    readBeat := 0.U
  }
  io.axi.r.valid     := readValid
  io.axi.r.bits.id   := read.id
  io.axi.r.bits.data := readData
  io.axi.r.bits.resp := Mux(readError, 2.U, 0.U)
  io.axi.r.bits.last := readBeat === read.len
  when(io.axi.r.fire) {
    readValid                        := false.B
    read.addr                        := read.addr + (1.U << read.size)
    readBeat                         := readBeat + 1.U
    when(io.axi.r.bits.last)(reading := false.B)
  }

  val writing       = RegInit(false.B)
  val write         = Reg(new axi4.Address(p))
  val writeBeat     = RegInit(0.U(8.W))
  val writeValid    = RegInit(false.B)
  val writeData     = Reg(UInt(128.W))
  val writeMask     = Reg(UInt(16.W))
  val writeHalf     = RegInit(false.B)
  val writeError    = RegInit(false.B)
  val responseValid = RegInit(false.B)
  io.axi.aw.ready                   := !writing && !responseValid
  when(io.axi.aw.fire) {
    assert(io.axi.aw.bits.size <= 4.U && (io.axi.aw.bits.burst === 1.U || io.axi.aw.bits.len === 0.U))
    assert((io.axi.aw.bits.addr & ((1.U(17.W) << io.axi.aw.bits.size) - 1.U)) === 0.U)
    assert(!io.axi.aw.bits.lock && io.axi.aw.bits.region === 0.U && io.axi.aw.bits.cache === 15.U)
    write      := io.axi.aw.bits
    writing    := true.B
    writeBeat  := 0.U
    writeError := false.B
  }
  io.axi.w.ready                    := writing && !writeValid
  when(io.axi.w.fire) {
    assert(io.axi.w.bits.last === (writeBeat === write.len))
    val allowed = (("hffff".U(16.W) >> (16.U - (1.U << write.size))) << write.addr(3, 0))(15, 0)
    assert((io.axi.w.bits.strb & ~allowed) === 0.U)
    writeData  := io.axi.w.bits.data
    writeMask  := io.axi.w.bits.strb
    writeValid := true.B
    writeHalf  := false.B
  }
  io.axi.b.valid                    := responseValid
  io.axi.b.bits.id                  := write.id
  io.axi.b.bits.resp                := Mux(writeError, 2.U, 0.U)
  when(io.axi.b.fire)(responseValid := false.B)

  val occupied    = RegInit(false.B)
  val ownerWrite  = Reg(Bool())
  val preferWrite = RegInit(false.B)
  val wantsRead   = reading && !readValid
  val wantsWrite  = writeValid
  val selectWrite = wantsWrite && (!wantsRead || preferWrite)
  val mask        = Mux(writeHalf, writeMask(15, 8), writeMask(7, 0))
  val skip        = !occupied && selectWrite && !mask.orR
  cache.io.access.valid      := !occupied && (wantsRead || wantsWrite) && !skip
  cache.io.access.bits       := 0.U.asTypeOf(cache.io.access.bits)
  cache.io.access.bits.addr  := Mux(
    selectWrite,
    Cat(write.addr(p.addressBits - 1, 4), writeHalf, 0.U(3.W)),
    Cat(read.addr(p.addressBits - 1, 3), 0.U(3.W))
  )
  cache.io.access.bits.write := selectWrite
  cache.io.access.bits.data  := Mux(writeHalf, writeData(127, 64), writeData(63, 0))
  cache.io.access.bits.mask  := Mux(selectWrite, mask, "hff".U)
  when(cache.io.access.fire) {
    occupied    := true.B
    ownerWrite  := selectWrite
    preferWrite := !selectWrite
  }
  cache.io.result.ready      := occupied
  when(cache.io.result.fire) {
    occupied := false.B
    when(ownerWrite) {
      writeError := writeError || cache.io.result.bits.error
    }.otherwise {
      readData  := (cache.io.result.bits.line >> (read.addr(5, 4) << 7))(127, 0)
      readError := cache.io.result.bits.error
      readValid := true.B
    }
  }
  when(skip || (cache.io.result.fire && ownerWrite)) {
    when(writeHalf) {
      writeValid := false.B
      write.addr := write.addr + (1.U << write.size)
      writeBeat  := writeBeat + 1.U
      when(writeBeat === write.len) { writing := false.B; responseValid := true.B }
    }.otherwise(writeHalf  := true.B)
    when(skip)(preferWrite := false.B)
  }
}
