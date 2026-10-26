package framework.acc.memdomain.tomem

import framework.lib.arbiter.{Arbiter}

import framework.acc.frontend.configs.FrontendParam
import framework.acc.gpdomain.configs.GpDomainParam
import framework.acc.memdomain.configs.MemDomainParam
import framework.core.memory.access.preflight.configs.PmpParams
import framework.acc.memdomain.tomem.configs.{Params => TransportParam}

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.acc.gpdomain.ImagePort
import framework.acc.balldomain.mmio_bank.MetadataWrite
import framework.acc.memdomain.{BankPort, Request, Response}
import framework.core.memory.access.preflight.{MapQuery, MapResult}
import framework.ip.bus.axi4
import framework.acc.memdomain.toddr._
import framework.lib.queue.Queue

class ControlPort(transportParam: TransportParam, permissionParam: PmpParams) extends Bundle {

  val read = Output(Valid(new DmaReadCommand(
    transportParam,
    permissionParam
  )))

  val write = Output(Valid(new WriteTransfer(
    transportParam,
    permissionParam
  )))

  val finished = Output(Vec(3, Valid(new DmaStatus)))
  val complete = Flipped(Vec(3, Decoupled(new DmaStatus)))

  val decisions = Input(Vec(
    3,
    new DmaDecision(transportParam)
  ))

  val queries = Output(Vec(
    2,
    new MapQuery(transportParam.mapping)
  ))

  val mappings = Input(Vec(
    2,
    new MapResult(transportParam.mapping)
  ))

  val coherence =
    new framework.acc.memdomain.tol1.Port(transportParam.interlock)

}

class Port(transportParam: TransportParam, permissionParam: PmpParams) extends Bundle {
  val control = new ControlPort(transportParam, permissionParam)

  val axi = new axi4.Port(transportParam.axi)

}

@instantiable
class ToMem(
  val frontendParam:   FrontendParam,
  val memParam:        MemDomainParam,
  val gpParam:         GpDomainParam,
  val transportParam:  TransportParam,
  val permissionParam: PmpParams)
    extends Module {
  val ownerId  = frontendParam.ownerId
  require(
    memParam.bankWidth == 128 && memParam.dma_buswidth == 128
  )
  val rowBytes = memParam.bankWidth / 8

  @public
  val io = IO(new Bundle {
    val request     = Flipped(Decoupled(new Request(frontendParam, memParam, permissionParam)))
    val response    = Decoupled(new Response(frontendParam))
    val bankRead    = new BankPort(frontendParam, memParam)
    val bankWrite   = new BankPort(frontendParam, memParam)
    val memory      = new framework.acc.memdomain.tomem.Port(transportParam, permissionParam)
    val metadata    = Flipped(new MetadataWrite(memParam))
    val image       = if (gpParam.enable) Some(new ImagePort) else None
    val queryVbank  = Output(UInt(frontendParam.bank_id_len.W))
    val queryShared = Output(Bool())
    val queryGroups = Input(UInt(memParam.groupCountWidth.W))
  })

  io.bankRead.thread_id        := ownerId.U
  io.bankWrite.thread_id       := ownerId.U
  io.bankRead.write.req.valid  := false.B
  io.bankRead.write.req.bits   := 0.U.asTypeOf(io.bankRead.write.req.bits)
  io.bankRead.write.resp.ready := true.B
  io.bankWrite.read.req.valid  := false.B
  io.bankWrite.read.req.bits   := 0.U.asTypeOf(io.bankWrite.read.req.bits)
  io.bankWrite.read.resp.ready := true.B

  val toDDR: Instance[ToDDR] =
    Instantiate(new ToDDR(transportParam, permissionParam))

  val toL1: Instance[framework.acc.memdomain.tol1.ToL1] =
    Instantiate(
      new framework.acc.memdomain.tol1.ToL1(transportParam.interlock)
    )

  val dma =
    Wire(new DmaPort(transportParam, permissionParam))
  toDDR.io.dma <> dma
  toDDR.io.decisions            := io.memory.control.decisions
  io.memory.control.queries     := toDDR.io.queries
  toDDR.io.mappings             := io.memory.control.mappings
  io.memory.axi <> toDDR.io.axi
  io.memory.control.coherence <> toL1.io
  io.memory.control.read.bits   := dma.read.bits
  io.memory.control.write.valid := dma.write.valid
  io.memory.control.write.bits  := dma.write.bits

  val idle :: query :: queryWait :: prepare :: Nil                                                                         = Enum(4)
  val setup                                                                                                                = RegInit(idle)
  val request                                                                                                              = Reg(new Request(frontendParam, memParam, permissionParam))
  val readIdle :: readOffer :: readReceive :: readBank :: readBankWait :: readTerminal :: imageWait :: readComplete :: Nil =
    Enum(8)
  val readState                                                                                                            = RegInit(readIdle)
  val writeIdle :: writeOffer :: writeStream :: writeWait :: writeDrain :: writeComplete :: Nil                            = Enum(6)
  val writeState                                                                                                           = RegInit(writeIdle)
  val read                                                                                                                 = Reg(new Request(frontendParam, memParam, permissionParam))
  val write                                                                                                                = Reg(new Request(frontendParam, memParam, permissionParam))
  val readGroups                                                                                                           = Reg(UInt(memParam.groupCountWidth.W))
  val writeGroups                                                                                                          = Reg(UInt(memParam.groupCountWidth.W))
  val readGroupBase                                                                                                        = Reg(UInt(memParam.groupIdWidth.W))
  val writeGroupBase                                                                                                       = Reg(UInt(memParam.groupIdWidth.W))
  val readGroup                                                                                                            = Reg(UInt(memParam.groupIdWidth.W))
  val writeGroup                                                                                                           = Reg(UInt(memParam.groupIdWidth.W))
  val readRow                                                                                                              = Reg(UInt(16.W))
  val writeRow                                                                                                             = Reg(UInt(16.W))
  val readWords                                                                                                            = Reg(UInt(32.W))
  val readData                                                                                                             = Reg(UInt(128.W))
  val readLast                                                                                                             = Reg(Bool())
  val imageWord                                                                                                            = Reg(UInt(2.W))
  val imageBytes                                                                                                           = Reg(UInt(32.W))
  val imageStarted                                                                                                         = RegInit(false.B)
  val writeFault                                                                                                           = Reg(new DmaStatus)
  val readFault                                                                                                            = Reg(new DmaStatus)
  val writeCursor                                                                                                          = Reg(UInt(64.W))
  val writeFinal                                                                                                           = Reg(Bool())
  val readsFinished                                                                                                        = RegInit(false.B)

  val wantsWrite = io.request.bits.cmd.is_store
  io.request.ready                := setup === idle && Mux(wantsWrite, writeState === writeIdle, readState === readIdle)
  when(io.request.fire) { request := io.request.bits; setup := query }
  io.queryVbank                   := request.cmd.bank_id
  io.queryShared                  := request.cmd.is_shared
  when(setup === query)(setup     := queryWait)
  when(setup === queryWait)(setup := prepare)

  val selected = request.cmd.special(63) && !request.cmd.is_mvin_2d && !request.cmd.is_image && !request.cmd.is_mmio
  val groups   =
    Mux(request.cmd.is_image || request.cmd.is_config || request.cmd.is_mmio, 1.U, Mux(selected, 1.U, io.queryGroups))
  val base     = Mux(selected, request.cmd.special(62, 58), 0.U)
  when(setup === prepare) {
    setup := idle
    when(request.cmd.is_image || request.cmd.is_config) {
      assert(
        gpParam.enable.B && request.cmd.bank_id >= memParam.virtualBankCount.U &&
          request.cmd.bank_id < (memParam.virtualBankCount + 2).U,
        "Program bank is outside the image slots"
      )
    }
    when(request.cmd.is_store) {
      writeFault     := 0.U.asTypeOf(new DmaStatus)
      write          := request
      writeGroups    := groups
      writeGroupBase := base
      writeGroup     := 0.U
      writeRow       := 0.U
      writeCursor    := request.cmd.mem_addr

      readsFinished := false.B
      writeFinal    := false.B
      writeState    := writeOffer
    }.otherwise {
      read          := request
      readGroups    := groups
      readGroupBase := base
      readGroup     := 0.U
      readRow       := Mux(request.cmd.is_mvin_2d, request.cmd.special(58, 53), 0.U)
      readWords     := 0.U
      imageWord     := 0.U
      imageBytes    := request.cmd.iter
      imageStarted  := false.B
      readFault     := 0.U.asTypeOf(readFault)
      readState     := Mux(request.cmd.is_config, readComplete, readOffer)
    }
    when(!request.cmd.is_image && !request.cmd.is_config && !request.cmd.is_mmio) {
      assert(groups =/= 0.U, "Memory stream requires an allocated bank")
      when(selected)(assert(base < io.queryGroups, "Selected group exceeds allocation"))
    }
  }

  val imageRead = read.cmd.is_image

  val imageRejected =
    if (gpParam.enable)
      io.image.get.complete.valid && io.image.get.complete.bits.fault
    else false.B

  val width  = read.cmd.special(61, 59) +& 1.U
  val beats  =
    Mux(imageRead, (imageBytes.pad(33) + 15.U) >> 4, read.cmd.iter * Mux(read.cmd.is_mvin_2d, width, readGroups))
  val stride = Mux(imageRead || read.cmd.is_mvin_2d || read.cmd.is_mmio, 1.U, read.cmd.special(57, 39))

  val readOfferReady =
    if (gpParam.enable) !imageRead || imageStarted && !imageRejected
    else true.B

  dma.read.valid                      := readState === readOffer && readOfferReady
  dma.read.bits.producer              := Mux(imageRead, 2.U, 0.U)
  dma.read.bits.transfer.vaddr        := read.cmd.mem_addr
  dma.read.bits.transfer.len          := Mux(imageRead, imageBytes, beats << 4)
  dma.read.bits.transfer.groups       := readGroups
  dma.read.bits.transfer.stride       := stride
  dma.read.bits.transfer.is_2d        := read.cmd.is_mvin_2d
  dma.read.bits.transfer.pixel_bytes  := read.cmd.special(42, 36) << 3
  dma.read.bits.transfer.source_width := read.cmd.special(52, 43)
  dma.read.bits.transfer.core_width   := width
  when(dma.read.fire)(readState       := readReceive)
  dma.readResult.ready                := readState === readReceive
  when(dma.readResult.fire) {
    readData              := dma.readResult.bits.data
    readLast              := dma.readResult.bits.last
    imageWord             := 0.U
    when(dma.readResult.bits.fault.error =/= DmaError.None.U || readFault.error =/= DmaError.None.U || imageRejected) {
      when(dma.readResult.bits.fault.error =/= DmaError.None.U)(readFault        := dma.readResult.bits.fault)
      when(imageRejected && readFault.error === DmaError.None.U)(readFault.error := DmaError.Cancelled.U)
      when(dma.readResult.bits.last)(readState                                   := Mux(imageRead, readTerminal, readComplete))
    }.otherwise(readState := readBank)
  }

  io.bankWrite.bank_id             := read.cmd.bank_id
  io.bankWrite.rob_id              := read.rob_id
  io.bankWrite.group_id            := readGroupBase + readGroup
  io.bankWrite.write.req.valid     := readState === readBank && !imageRead && !read.cmd.is_mmio
  io.bankWrite.write.req.bits.addr := readRow
  // MVIN_2D replaces the entire row, including zero-padding unused bytes.
  io.bankWrite.write.req.bits.data :=
    Mux(read.cmd.is_mvin_2d && read.cmd.special(62), Cat(0.U(64.W), readData(63, 0)), readData)
  for (i <- 0 until 16) io.bankWrite.write.req.bits.mask(i) := true.B
  io.bankWrite.write.resp.ready                                           := readState === readBankWait && !read.cmd.is_mmio
  io.metadata.request.valid                                               := readState === readBank && read.cmd.is_mmio
  io.metadata.request.bits.addr                                           := read.cmd.special(55, 39) + (readRow << 4)
  io.metadata.request.bits.data                                           := readData
  io.metadata.request.bits.mask                                           := ((1.U(17.W) << read.cmd.special(63, 56)) - 1.U)(15, 0)
  io.metadata.response.ready                                              := readState === readBankWait && read.cmd.is_mmio
  when(io.metadata.request.fire) {
    assert(memParam.mmioEnable.B && read.cmd.special(63, 56) >= 1.U && read.cmd.special(63, 56) <= 16.U)
    assert(read.cmd.special(55, 39) +& (read.cmd.iter << 4) <= memParam.mmioTotalBytes.U)
  }
  when(io.bankWrite.write.req.fire || io.metadata.request.fire)(readState := readBankWait)
  when(io.bankWrite.write.resp.fire || io.metadata.response.fire) {
    when(Mux(read.cmd.is_mmio, !io.metadata.response.bits, !io.bankWrite.write.resp.bits.ok)) {
      readFault.error := DmaError.Bank.U; readFault.address := read.cmd.mem_addr
    }
    when(readGroup +& 1.U === readGroups || read.cmd.is_mvin_2d) {
      readGroup := 0.U
      readRow   := readRow + 1.U
    }.otherwise(readGroup := readGroup + 1.U)
    readState             := Mux(readLast, readComplete, readReceive)
  }

  io.image.foreach { image =>
    image.start.valid                   := readState === readOffer && imageRead && !imageStarted
    image.start.bits.buffer             := (read.cmd.bank_id - memParam.virtualBankCount.U)(
      0
    )
    image.start.bits.bytes              := imageBytes
    when(image.start.fire)(imageStarted := true.B)
    image.data.valid                    := readState === readBank && imageRead && !imageRejected
    image.data.bits                     := (readData >> (imageWord << 5))(31, 0)
    when(image.data.fire) {
      readWords := readWords + 1.U
      imageWord := imageWord + 1.U
      when((readWords +& 1.U) * 4.U >= imageBytes || imageWord === 3.U) {
        readState := Mux(readLast, readTerminal, readReceive)
      }
    }
    when(imageRead && imageStarted && imageRejected) {
      when(readState === readOffer) {
        readFault.error := DmaError.Cancelled.U
        readState       := readTerminal
      }
      when(readState === readBank) {
        readFault.error := DmaError.Cancelled.U
        readState       := Mux(readLast, readTerminal, readReceive)
      }
    }
    image.terminal.valid                := readState === readTerminal
    image.terminal.bits                 := readFault
    when(image.terminal.fire)(readState := imageWait)
    image.complete.ready                := readState === imageWait
    when(image.complete.fire) {
      when(image.complete.bits.fault && (readFault.error === DmaError.None.U || readFault.error === DmaError.Cancelled.U)) {
        readFault.error   := Mux(
          image.complete.bits.cause === 5.U || image.complete.bits.cause === 7.U,
          DmaError.Bank.U,
          DmaError.Shape.U
        )
        readFault.address := image.complete.bits.tval
      }
      readState := readComplete
    }
    image.release.valid                 := setup === prepare && request.cmd.is_config
    image.release.bits                  := (request.cmd.bank_id - memParam.virtualBankCount.U)(
      0
    )
  }

  val tags = Instantiate(new Queue(Bool(), 2, false, true))

  val data = Instantiate(new Queue(
    new WriteData(transportParam),
    2,
    true,
    false
  ))

  // Accepted SRAM reads reserve a slot until their data leaves the staging queue.
  val reserved = tags.io.count +& data.io.count

  val totalBytes      = write.cmd.iter * writeGroups * rowBytes.U
  val writeStride     = write.cmd.special(57, 39)
  val descriptorBytes = Mux(writeStride === 1.U, totalBytes, writeGroups * rowBytes.U)
  dma.write.valid      := writeState === writeOffer
  dma.write.bits.vaddr := writeCursor
  dma.write.bits.len   := descriptorBytes
  when(dma.write.fire) {
    writeFinal := writeStride === 1.U || writeRow +& 1.U === write.cmd.iter
    writeState := writeStream
  }
  val lastGroup = writeGroup +& 1.U === writeGroups
  val lastRow = writeRow +& 1.U === write.cmd.iter
  val last    = lastGroup && (writeStride =/= 1.U || lastRow)
  io.bankRead.inst_id                                            := write.instId
  io.bankWrite.inst_id                                           := read.instId
  io.bankRead.bank_id                                            := write.cmd.bank_id
  io.bankRead.rob_id                                             := write.rob_id
  io.bankRead.group_id                                           := writeGroupBase + writeGroup
  io.bankRead.read.req.valid                                     := writeState === writeStream && !readsFinished && reserved < 2.U && tags.io.enq.ready
  io.bankRead.read.req.bits.addr                                 := writeRow
  tags.io.enq.valid                                              := io.bankRead.read.req.fire
  tags.io.enq.bits                                               := last
  io.bankRead.read.resp.ready                                    := data.io.enq.ready && tags.io.deq.valid
  data.io.enq.valid                                              := io.bankRead.read.resp.valid && tags.io.deq.valid
  data.io.enq.bits.data                                          := io.bankRead.read.resp.bits.data
  data.io.enq.bits.last                                          := tags.io.deq.bits
  tags.io.deq.ready                                              := io.bankRead.read.resp.fire
  when(io.bankRead.read.req.fire) {
    when(lastGroup) { writeGroup := 0.U; writeRow := writeRow + 1.U }
      .otherwise(writeGroup  := writeGroup + 1.U)
    when(last)(readsFinished := true.B)
  }
  dma.writeData.valid                                            := writeState === writeStream && data.io.deq.valid
  dma.writeData.bits                                             := data.io.deq.bits
  data.io.deq.ready                                              := writeState === writeStream && dma.writeData.ready || writeState === writeDrain
  when(dma.writeData.fire && dma.writeData.bits.last)(writeState := writeWait)
  dma.writeResult.ready                                          := writeState === writeWait
  when(dma.writeResult.fire) {
    when(dma.writeResult.bits.fault.error =/= DmaError.None.U) {
      writeFault := dma.writeResult.bits.fault

      writeState := writeDrain
    }.elsewhen(writeFinal)(writeState := writeComplete)
      .otherwise {
        writeCursor   := writeCursor + writeGroups * writeStride * rowBytes.U
        readsFinished := false.B
        writeState    := writeOffer
      }
  }
  when(writeState === writeDrain && reserved === 0.U)(writeState := writeComplete)

  val readProducer = Mux(imageRead, 2.U, 0.U)
  for (i <- 0 until 3) {
    io.memory.control.finished(i).valid := (if (i == 1) writeState === writeComplete
                                            else
                                              readState === readComplete && !read.cmd.is_config && readProducer === i.U)
    io.memory.control.finished(i).bits  := (if (i == 1) writeFault else readFault)
    io.memory.control.complete(i).ready := false.B
  }
  val completion = Instantiate(new Arbiter(new Response(frontendParam), 2))
  completion.io.in(0).valid                 := readState === readComplete &&
    (read.cmd.is_config || io.memory.control.complete(readProducer).valid)
  completion.io.in(0).bits.rob_id           := read.rob_id
  completion.io.in(0).bits.fault            := Mux(read.cmd.is_config, readFault, io.memory.control.complete(readProducer).bits)
  completion.io.in(1).valid                 := writeState === writeComplete && io.memory.control.complete(1).valid
  completion.io.in(1).bits.rob_id           := write.rob_id
  completion.io.in(1).bits.fault            := io.memory.control.complete(1).bits
  io.memory.control.complete(
    0
  ).ready                                   := readState === readComplete && !imageRead && !read.cmd.is_config && completion.io.in(0).ready
  io.memory.control.complete(
    2
  ).ready                                   := readState === readComplete && imageRead && !read.cmd.is_config && completion.io.in(0).ready
  io.memory.control.complete(1).ready       := writeState === writeComplete && completion.io.in(1).ready
  io.response <> completion.io.out
  when(completion.io.in(0).fire)(readState  := readIdle)
  when(completion.io.in(1).fire)(writeState := writeIdle)
  io.bankRead.is_shared                     := write.cmd.is_shared
  io.bankWrite.is_shared                    := read.cmd.is_shared

  // Publish the read shape before image-loader backpressure can delay DMA.
  // The setup stage accepts streams in order; coherence reservation follows that order.
  io.memory.control.read.valid := readState === readOffer
  val readAccess = dma.read.bits.access
  readAccess.context                 := read.context
  readAccess.baseVA                  := read.cmd.mem_addr
  readAccess.rows                    := Mux(imageRead, 1.U, read.cmd.iter)
  readAccess.columns                 := Mux(read.cmd.is_mvin_2d, width, 1.U)
  readAccess.spanBytes               := Mux(imageRead, (imageBytes.pad(64) + 15.U) & ~15.U(64.W), readGroups * rowBytes.U)
  readAccess.rowStride               := Mux(
    read.cmd.is_mvin_2d,
    read.cmd.special(52, 43) * (read.cmd.special(42, 36) << 3),
    readGroups * stride * rowBytes.U
  )
  readAccess.columnStride            := Mux(read.cmd.is_mvin_2d, read.cmd.special(42, 36) << 3, 0.U)
  dma.write.bits.access.context      := write.context
  dma.write.bits.access.baseVA       := write.cmd.mem_addr
  dma.write.bits.access.rows         := write.cmd.iter
  dma.write.bits.access.columnStride := 0.U
  dma.write.bits.access.columns      := 1.U
  dma.write.bits.access.spanBytes    := writeGroups * rowBytes.U
  dma.write.bits.access.rowStride    := writeGroups * writeStride * rowBytes.U
}
