package framework.memdomain.frontend.mem.dma

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.top.GlobalConfig
import memcore.memory.preflight.{Error, MapQuery, MapResult, Params}
import memcore.bus.axi4

@instantiable
class WriteDma(b: GlobalConfig, prepared: Params, axiParams: axi4.Params) extends Module {
  require(b.memDomain.bankWidth == 128 && b.memDomain.dma_buswidth == 128 && axiParams.dataBits == 128)
  require(prepared.beatBytes == 16 && prepared.bus.addressBits == axiParams.addressBits)

  @public
  val io = IO(new Bundle {
    val req           = Flipped(Decoupled(new BBWriteCommand))
    val data          = Flipped(Decoupled(new BBWriteData(128)))
    val resp          = Decoupled(new BBWriteResponse)
    val parentId      = Input(UInt(prepared.idBits.W))
    val decisionValid = Input(Bool())
    val decisionFault = Input(new DmaStatus)
    val busy          = Output(Bool())
    val query         = Output(new MapQuery(prepared))
    val mapping       = Input(new MapResult(prepared))
    val axi           = new axi4.Port(axiParams)
  })

  val idle :: plan :: query :: stream :: waitB :: drain :: respond :: Nil = Enum(7)
  val state                                                               = RegInit(idle)
  val command                                                             = Reg(new BBWriteCommand)
  val parent                                                              = Reg(UInt(prepared.idBits.W))
  val endVA                                                               = Reg(UInt(64.W))
  val cursorVA                                                            = Reg(UInt(64.W))
  val inputTotal                                                          = Reg(UInt(29.W))
  val consumed                                                            = RegInit(0.U(29.W))
  val sourceEnded                                                         = RegInit(false.B)
  val remaining                                                           = Reg(UInt(29.W))
  val firstBeat                                                           = RegInit(true.B)
  val beats                                                               = Reg(UInt(9.W))
  val beatIndex                                                           = RegInit(0.U(9.W))
  val awSent                                                              = RegInit(false.B)
  val wDone                                                               = RegInit(false.B)
  val queryVA                                                             = Reg(UInt(64.W))
  val queryBytes                                                          = Reg(UInt(13.W))
  val headOffset                                                          = Reg(UInt(4.W))
  val physical                                                            = Reg(UInt(axiParams.addressBits.W))
  val carry                                                               = RegInit(0.U(128.W))
  val outputValid                                                         = RegInit(false.B)
  val outputData                                                          = Reg(UInt(128.W))
  val outputMask                                                          = Reg(UInt(16.W))
  val outputLast                                                          = Reg(Bool())
  val fault                                                               = RegInit(0.U.asTypeOf(new DmaStatus))
  val offset                                                              = command.vaddr(3, 0)
  val expectedLast                                                        = consumed +& 1.U === inputTotal
  val outputRoom                                                          = !outputValid || (io.axi.w.ready && !outputLast)
  val outputIndex                                                         = beatIndex + io.axi.w.fire.asUInt
  val outputHead                                                          = firstBeat && !io.axi.w.fire

  io.req.ready         := state === idle && io.decisionValid
  io.busy              := state =/= idle
  io.data.ready        := (state === stream && !wDone && outputRoom && fault.error === DmaError.None.U && consumed < inputTotal) ||
    (state === drain && !sourceEnded && consumed < inputTotal)
  io.resp.valid        := state === respond
  io.resp.bits.done    := fault.error === DmaError.None.U
  io.resp.bits.fault   := fault
  io.query             := 0.U.asTypeOf(new MapQuery(prepared))
  io.query.valid       := state === query
  io.query.id          := parent
  io.query.va          := queryVA
  io.query.bytes       := queryBytes
  io.query.write       := true.B
  io.axi.aw.valid      := state === stream && !awSent
  io.axi.aw.bits       := 0.U.asTypeOf(new axi4.Address(axiParams))
  io.axi.aw.bits.id    := 1.U
  io.axi.aw.bits.addr  := physical
  io.axi.aw.bits.len   := beats - 1.U
  io.axi.aw.bits.size  := 4.U
  io.axi.aw.bits.burst := 1.U
  io.axi.w.valid       := state === stream && outputValid
  io.axi.w.bits.data   := outputData
  io.axi.w.bits.strb   := outputMask
  io.axi.w.bits.last   := outputLast
  io.axi.b.ready       := state === waitB
  io.axi.ar.valid      := false.B
  io.axi.ar.bits       := 0.U.asTypeOf(new axi4.Address(axiParams))
  io.axi.r.ready       := false.B

  def terminate(error: UInt, at: UInt): Unit = {
    fault.error   := error
    fault.address := at
    state         := Mux(sourceEnded || consumed === inputTotal, respond, drain)
  }

  when(io.req.fire) {
    command  := io.req.bits; parent := io.parentId
    consumed := 0.U; sourceEnded    := false.B; firstBeat := true.B
    carry    := 0.U; outputValid    := false.B; fault     := 0.U.asTypeOf(new DmaStatus)
    val finalByte = io.req.bits.vaddr +& (io.req.bits.len - 1.U)
    val malformed = io.req.bits.len === 0.U || io.req.bits.len(3, 0) =/= 0.U || finalByte(64)
    inputTotal        := io.req.bits.len >> 4
    remaining         := (io.req.bits.len >> 4) +& (io.req.bits.vaddr(3, 0) =/= 0.U).asUInt
    cursorVA          := io.req.bits.vaddr & ~15.U(64.W)
    endVA             := finalByte(63, 0)
    when(malformed) {
      fault.error := DmaError.Shape.U; fault.address := io.req.bits.vaddr; state := respond
    }.elsewhen(io.decisionFault.error =/= DmaError.None.U) {
      fault := io.decisionFault; state := drain
    }.otherwise(state := plan)
  }

  when(state === plan) {
    val pageBeats    = (4096.U(13.W) - cursorVA(11, 0)) >> 4
    val capped       = Mux(remaining < 256.U, remaining, 256.U)
    val extent       = Mux(capped < pageBeats, capped, pageBeats)
    val head         = Mux(firstBeat, offset, 0.U)
    val start        = cursorVA + head
    val covered      = (extent << 4) - head
    val commandBytes = endVA - start + 1.U
    beats      := extent; beatIndex := 0.U; awSent := false.B; wDone := false.B
    headOffset := head; queryVA     := start
    queryBytes := Mux(commandBytes < covered, commandBytes, covered)
    state      := query
  }
  when(state === query) {
    val base    = io.mapping.pa.pad(axiParams.addressBits + 1) - headOffset
    val lastPA  = base +& ((beats << 4) - 1.U)
    val invalid = !io.mapping.hit || io.mapping.error =/= Error.Ok.U ||
      base(axiParams.addressBits) || base(3, 0) =/= 0.U ||
      lastPA(lastPA.getWidth - 1, axiParams.addressBits).orR ||
      (base >> 12) =/= (lastPA >> 12)
    when(invalid) {
      val error = MuxLookup(io.mapping.error, DmaError.AccessFault.U)(Seq(
        Error.Shape.U     -> DmaError.Shape.U,
        Error.Overflow.U  -> DmaError.Shape.U,
        Error.PageFault.U -> DmaError.PageFault.U,
        Error.Capacity.U  -> DmaError.Capacity.U,
        Error.Context.U   -> DmaError.Context.U
      ))
      terminate(error, queryVA)
    }.otherwise { physical := base(axiParams.addressBits - 1, 0); state := stream }
  }
  when(io.axi.aw.fire)(awSent := true.B)

  // An accepted old W can be replaced immediately; refills take priority over clearing valid.
  when(io.axi.w.fire)(outputValid                                         := false.B)
  when(io.data.fire) {
    val properLast = io.data.bits.last === expectedLast
    assert(properLast, "WriteDma input LAST does not match command length")
    consumed                                            := consumed + 1.U
    when(io.data.bits.last || expectedLast)(sourceEnded := true.B)
    when(state === stream) {
      outputData  := Mux(
        properLast,
        Mux(offset === 0.U, io.data.bits.data, carry | (io.data.bits.data << (offset * 8.U))(127, 0)),
        0.U
      )
      outputMask  := Mux(properLast, Mux(outputHead, ("hffff".U(16.W) << offset)(15, 0), "hffff".U), 0.U)
      outputLast  := outputIndex +& 1.U === beats
      outputValid := true.B
      carry       := (io.data.bits.data >> ((16.U - offset) * 8.U))(127, 0)
      when(!properLast && fault.error === DmaError.None.U) {
        fault.error   := DmaError.Protocol.U
        fault.address := command.vaddr + (consumed << 4)
      }
    }
  }
  when(state === stream && !wDone && outputRoom) {
    when(fault.error =/= DmaError.None.U) {
      // Complete an already accepted AW even if the producer ended early.
      outputData := 0.U; outputMask                           := 0.U
      outputLast := outputIndex +& 1.U === beats; outputValid := true.B
    }.elsewhen(consumed === inputTotal) {
      outputData := carry
      outputMask := ((1.U(17.W) << offset) - 1.U)(15, 0)
      outputLast := outputIndex +& 1.U === beats; outputValid := true.B
    }
  }
  when(io.axi.w.fire) {
    firstBeat              := false.B
    cursorVA               := cursorVA + 16.U; remaining := remaining - 1.U
    beatIndex              := beatIndex + 1.U
    when(outputLast)(wDone := true.B)
  }
  when(state === stream && (awSent || io.axi.aw.fire) &&
    (wDone || (io.axi.w.fire && outputLast)))(state                       := waitB)
  when(io.axi.b.fire) {
    val bad = io.axi.b.bits.id =/= 1.U || io.axi.b.bits.resp =/= 0.U
    when(bad && fault.error === DmaError.None.U) {
      fault.error   := Mux(
        io.axi.b.bits.id =/= 1.U || io.axi.b.bits.resp === 1.U,
        DmaError.Protocol.U,
        DmaError.AccessFault.U
      )
      fault.address := physical
    }
    when(bad || fault.error =/= DmaError.None.U) {
      state := Mux(sourceEnded || consumed === inputTotal, respond, drain)
    }.elsewhen(remaining === 0.U)(state := respond).otherwise(state := plan)
  }
  when(state === drain && (sourceEnded || consumed === inputTotal))(state := respond)
  when(io.resp.fire)(state                                                := idle)
}
