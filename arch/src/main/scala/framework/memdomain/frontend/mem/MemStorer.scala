package framework.memdomain.frontend.mem

import memcore.memory.queue.Queue

import chisel3._
import chisel3.util._
import framework.top.GlobalConfig
import framework.memdomain.frontend.cmd.rs.{MemRsComplete, MemRsIssue}
import framework.memdomain.frontend.mem.dma.{BBWriteCommand, BBWriteData, BBWriteResponse, DmaError, DmaStatus}
import framework.balldomain.blink.BankRead
import chisel3.experimental.hierarchy.{instantiable, public}

@instantiable
class MemStorer(val b: GlobalConfig) extends Module {
  require(
    b.memDomain.bankWidth == 128 && b.memDomain.dma_buswidth == 128,
    "MemStorer requires 128-bit bank and DMA beats"
  )
  val lineBytes = b.memDomain.bankWidth / 8
  val lgLine    = log2Ceil(lineBytes)
  val robBits   = log2Up(b.frontend.rob_entries)

  @public
  val io = IO(new Bundle {
    val cmdReq            = Flipped(Decoupled(new MemRsIssue(b)))
    val cmdResp           = Decoupled(new MemRsComplete(b))
    val footprint         = Output(new Footprint(b))
    val dmaReq            = Decoupled(new BBWriteCommand)
    val dmaData           = Decoupled(new BBWriteData(b.memDomain.bankWidth))
    val dmaResp           = Flipped(Decoupled(new BBWriteResponse))
    val bankRead          = Flipped(new BankRead(b))
    val query_valid       = Output(Bool())
    val query_vbank_id    = Output(UInt(b.memDomain.vbankIdWidth.W))
    val query_is_shared   = Output(Bool())
    val query_group_count = Input(UInt(b.memDomain.groupCountWidth.W))
    val is_shared         = Output(Bool())
  })

  val idle :: setup :: query :: queryWait :: prepare :: descriptor :: stream :: waitResponse :: drain :: done :: Nil =
    Enum(10)
  val state                                                                                                          = RegInit(idle)
  val rob                                                                                                            = RegInit(0.U(robBits.W))
  val isSub                                                                                                          = RegInit(false.B)
  val subRob                                                                                                         = RegInit(0.U(log2Up(b.frontend.sub_rob_depth * 4).W))
  val iterations                                                                                                     = RegInit(0.U(b.frontend.iter_len.W))
  val bank                                                                                                           = RegInit(0.U(b.memDomain.vbankIdWidth.W))
  val stride                                                                                                         = RegInit(1.U(19.W))
  val groups                                                                                                         = RegInit(1.U(b.memDomain.groupCountWidth.W))
  val shared                                                                                                         = RegInit(false.B)
  val baseVA                                                                                                         = RegInit(0.U(64.W))
  val cursor                                                                                                         = RegInit(0.U(64.W))
  val fault                                                                                                          = RegInit(0.U.asTypeOf(new DmaStatus))
  val footprintValid                                                                                                 = RegInit(false.B)
  val readRow                                                                                                        = RegInit(0.U(b.frontend.iter_len.W))
  val readGroup                                                                                                      = RegInit(0.U(b.memDomain.groupCountWidth.W))
  val readsFinished                                                                                                  = RegInit(false.B)
  val finalDescriptor                                                                                                = RegInit(false.B)

  val selectedGroup = RegInit(false.B)
  val groupBase     = RegInit(0.U(5.W))

  // Each accepted bank read reserves a data slot until its beat is sent.
  // Ordered last tags include requests whose synchronous response has not yet arrived.
  val lastTags = Module(new Queue(Bool(), 2, flow = true))
  val data     = Module(new Queue(new BBWriteData(b.memDomain.bankWidth), 2, pipe = true))
  val reserved = lastTags.io.count +& data.io.count
  val empty    = lastTags.io.count === 0.U && data.io.count === 0.U

  val span            = groups * lineBytes.U
  val rowStride       = groups * stride * lineBytes.U
  val totalBytes      = iterations * span
  val descriptorBytes = Mux(stride === 1.U, totalBytes, span)
  val shapeEnd        = baseVA +& ((iterations - 1.U) * rowStride) +& (span - 1.U)

  val strideOverflow =
    if (rowStride.getWidth > b.memDomain.memAddrLen)
      (rowStride >> b.memDomain.memAddrLen).orR
    else false.B

  val lengthOverflow = if (descriptorBytes.getWidth > 32) (descriptorBytes >> 32).orR else false.B
  val storageRows    = Mux(shared, b.memDomain.sharedBankEntries.U, b.memDomain.bankEntries.U)

  val shapeError = iterations === 0.U || iterations > storageRows ||
    groups === 0.U || groups > (BigInt(1) << b.memDomain.groupIdWidth).U || stride === 0.U ||
    (shapeEnd >> b.memDomain.memAddrLen).orR || strideOverflow || lengthOverflow

  io.cmdReq.ready             := state === idle
  when(io.cmdReq.fire) {
    assert(io.cmdReq.bits.cmd.is_store, "MemStorer received a non-store command")
    assert(empty, "MemStorer reused a command with pending bank reads or data")
    rob            := io.cmdReq.bits.rob_id
    isSub          := io.cmdReq.bits.is_sub
    subRob         := io.cmdReq.bits.sub_rob_id
    iterations     := io.cmdReq.bits.cmd.iter
    bank           := io.cmdReq.bits.cmd.bank_id
    selectedGroup  := io.cmdReq.bits.cmd.special(63)
    groupBase      := Mux(io.cmdReq.bits.cmd.special(63), io.cmdReq.bits.cmd.special(62, 58), 0.U)
    stride         := io.cmdReq.bits.cmd.special(57, 39)
    shared         := io.cmdReq.bits.cmd.is_shared
    baseVA         := io.cmdReq.bits.cmd.mem_addr
    cursor         := io.cmdReq.bits.cmd.mem_addr
    readRow        := 0.U
    readGroup      := 0.U
    readsFinished  := false.B
    fault          := 0.U.asTypeOf(new DmaStatus)
    footprintValid := false.B
    state          := setup
  }
  io.query_valid              := state === setup || state === query || state === queryWait
  io.query_vbank_id           := bank
  io.query_is_shared          := shared && io.query_valid
  when(state === setup)(state := query)
  when(state === query)(state := queryWait)
  when(state === queryWait) {
    when(selectedGroup) {
      assert(groupBase < io.query_group_count, "MemStorer selected group is outside bank allocation")
    }
    groups := Mux(selectedGroup, 1.U, io.query_group_count)
    state  := prepare
  }
  when(state === prepare) {
    footprintValid    := true.B
    when(shapeError) {
      fault.error   := DmaError.Shape.U
      fault.address := baseVA
      state         := drain
    }.otherwise(state := descriptor)
  }

  io.footprint               := 0.U.asTypeOf(new Footprint(b))
  io.footprint.valid         := footprintValid
  io.footprint.rob_id        := rob
  io.footprint.is_sub        := isSub
  io.footprint.sub_rob_id    := subRob
  io.footprint.baseVA        := baseVA
  io.footprint.rows          := iterations
  io.footprint.columns       := 1.U
  io.footprint.spanBytes     := span
  io.footprint.columnStride  := 0.U
  io.footprint.rowStride     := rowStride
  io.footprint.write         := true.B
  io.footprint.fault.error   := Mux(shapeError, DmaError.Shape.U, DmaError.None.U)
  io.footprint.fault.address := Mux(shapeError, baseVA, 0.U)

  io.dmaReq.valid      := state === descriptor
  io.dmaReq.bits.vaddr := cursor
  io.dmaReq.bits.len   := descriptorBytes
  when(io.dmaReq.fire) {
    assert(empty, "MemStorer started a descriptor with pending bank reads or data")
    finalDescriptor := stride === 1.U || readRow === iterations - 1.U
    readsFinished   := false.B
    state           := stream
  }

  val lastGroup = readGroup === groups - 1.U
  val lastRow   = readRow === iterations - 1.U
  val lastRead  = lastGroup && (stride =/= 1.U || lastRow)
  io.bankRead.rob_id           := rob
  io.bankRead.bank_id          := bank
  io.bankRead.ball_id          := 0.U
  io.bankRead.group_id         := groupBase + readGroup
  io.bankRead.io.req.bits.addr := readRow
  io.is_shared                 := shared
  io.bankRead.io.req.valid     := state === stream && !readsFinished &&
    (reserved < 2.U || io.dmaData.fire)
  lastTags.io.enq.valid        := io.bankRead.io.req.fire
  lastTags.io.enq.bits         := lastRead
  when(io.bankRead.io.req.fire) {
    assert(lastTags.io.enq.ready, "MemStorer bank read has no reserved last-tag slot")
    when(lastGroup) {
      readGroup              := 0.U
      when(!lastRow)(readRow := readRow + 1.U)
    }.otherwise(readGroup        := readGroup + 1.U)
    when(lastRead)(readsFinished := true.B)
  }

  data.io.enq.valid         := io.bankRead.io.resp.valid && lastTags.io.deq.valid
  data.io.enq.bits.data     := io.bankRead.io.resp.bits.data
  data.io.enq.bits.last     := lastTags.io.deq.bits
  io.bankRead.io.resp.ready := data.io.enq.ready && lastTags.io.deq.valid
  lastTags.io.deq.ready     := io.bankRead.io.resp.fire
  when(io.bankRead.io.resp.valid) {
    assert(lastTags.io.deq.valid, "MemStorer received an unreserved bank response")
    assert(data.io.enq.ready, "MemStorer reserved bank response exceeded its data buffer")
  }
  assert(reserved <= 2.U, "MemStorer exceeded its two reserved data slots")

  io.dmaData.valid                                    := state === stream && data.io.deq.valid
  io.dmaData.bits                                     := data.io.deq.bits
  data.io.deq.ready                                   := state === drain || (state === stream && io.dmaData.ready)
  when(io.dmaData.fire && io.dmaData.bits.last)(state := waitResponse)

  // WriteDma consumes the complete descriptor even after an AXI/map error, and
  // returns exactly one terminal response. Do not advance a strided row before it.
  io.dmaResp.ready                     := state === waitResponse
  when(io.dmaResp.fire) {
    when(io.dmaResp.bits.fault.error =/= DmaError.None.U || !io.dmaResp.bits.done) {
      when(io.dmaResp.bits.fault.error =/= DmaError.None.U)(fault := io.dmaResp.bits.fault)
        .otherwise { fault.error := DmaError.Protocol.U; fault.address := cursor }
      state := drain
    }.elsewhen(finalDescriptor)(state := drain)
      .otherwise { cursor := cursor + rowStride; state := descriptor }
  }
  when(state === drain && empty)(state := done)
  io.cmdResp.valid                     := state === done
  io.cmdResp.bits                      := 0.U.asTypeOf(new MemRsComplete(b))
  io.cmdResp.bits.rob_id               := rob
  io.cmdResp.bits.is_sub               := isSub
  io.cmdResp.bits.sub_rob_id           := subRob
  io.cmdResp.bits.fault                := fault
  when(io.cmdResp.fire) { footprintValid := false.B; state := idle }
}
