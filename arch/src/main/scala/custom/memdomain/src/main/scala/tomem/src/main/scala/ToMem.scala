package framework.custom.memdomain.tomem

import framework.generic.acc.configs.BallLink
import framework.generic.contract.blink.BlinkParam

import framework.generic.acc.memdomain.tomem.configs.{Params => TransportParam}
import framework.generic.contract.memdomain.toddr.DmaError
import framework.custom.memdomain.toddr.DmaPort
import framework.custom.memdomain.toddr.DmaReadCommand
import framework.custom.memdomain.toddr.WriteTransfer
import framework.generic.arbiter.{Arbiter}

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import framework.custom.gpdomain.ImagePort
import framework.custom.memdomain.Request
import framework.custom.memdomain.toddr._
import framework.generic.contract.memory.access.preflight.configs.PmpParams
import framework.generic.acc.frontend.configs.FrontendParam
import framework.generic.acc.gpdomain.configs.GpDomainParam
import framework.generic.acc.memdomain.configs.MemDomainParam
import framework.generic.contract.mmio_bank.MetadataWrite
import framework.generic.contract.memdomain.BankPort
import framework.generic.contract.memdomain.Response
import framework.generic.contract.memdomain.toddr.DmaStatus

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

}

class Port(transportParam: TransportParam, permissionParam: PmpParams) extends Bundle {
  val control = new ControlPort(transportParam, permissionParam)

}

@instantiable
class ToMem(
  val frontendParam:   FrontendParam,
  val memParam:        MemDomainParam,
  val gpParam:         GpDomainParam,
  val transportParam:  TransportParam,
  val permissionParam: PmpParams)
    extends Module {
  require(
    memParam.bankWidth == 128 && memParam.dma_buswidth == 128
  )
  val rowBytes = memParam.bankWidth / 8

  @public
  val io = IO(new Bundle {
    // The thread (or hart) the accelerator serves: a strap, so that every
    // thread of a hart shares one module.
    val ownerId     = Input(UInt(64.W))
    val request     = Flipped(Decoupled(new Request(frontendParam, memParam, permissionParam)))
    val response    = Decoupled(new Response(log2Up(frontendParam.rob_entries)))
    val bankRead    = new BankPort(BallLink.blink(frontendParam, memParam))
    val bankWrite   = new BankPort(BallLink.blink(frontendParam, memParam))
    val memory      = new framework.custom.memdomain.tomem.Port(transportParam, permissionParam)
    val dma         = new DmaPort(transportParam, permissionParam)
    val metadata    = Flipped(new MetadataWrite(memParam.bankWidth, memParam.mmioTotalBytes))
    val image       = if (gpParam.enable) Some(new ImagePort) else None
    val queryVbank  = Output(UInt(frontendParam.bank_id_len.W))
    val queryShared = Output(Bool())
    val queryGroups = Input(UInt(memParam.groupCountWidth.W))
  })

  io.bankRead.thread_id        := io.ownerId
  io.bankWrite.thread_id       := io.ownerId
  io.bankRead.write.req.valid  := false.B
  io.bankRead.write.req.bits   := 0.U.asTypeOf(io.bankRead.write.req.bits)
  io.bankRead.write.resp.ready := true.B
  io.bankWrite.read.req.valid  := false.B
  io.bankWrite.read.req.bits   := 0.U.asTypeOf(io.bankWrite.read.req.bits)
  io.bankWrite.read.resp.ready := true.B

  val dma = io.dma
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
  val readBlockBase                                                                                                        = Reg(UInt(16.W))
  val readBlockRow                                                                                                         = Reg(UInt(5.W))
  val readBankRows                                                                                                         = RegInit(0.U(3.W))
  val writeDataRows                                                                                                        = Reg(UInt(18.W))
  val writeBankRows                                                                                                        = RegInit(0.U(3.W))
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
  val readRejected                                                                                                         = Reg(Bool())
  val writeRejected                                                                                                        = Reg(Bool())

  val wantsWrite = io.request.bits.cmd.is_store
  io.request.ready                := setup === idle && Mux(wantsWrite, writeState === writeIdle, readState === readIdle)
  when(io.request.fire) { request := io.request.bits; setup := query }
  io.queryVbank                   := request.cmd.bank_id
  io.queryShared                  := false.B
  when(setup === query)(setup     := queryWait)
  when(setup === queryWait)(setup := prepare)

  val selected = request.cmd.is_strided || (request.cmd.special(
    63
  ) && !request.cmd.is_mvin_2d && !request.cmd.is_image && !request.cmd.is_mmio)

  val groups    =
    Mux(request.cmd.is_image || request.cmd.is_config || request.cmd.is_mmio, 1.U, Mux(selected, 1.U, io.queryGroups))
  val base      = Mux(request.cmd.is_strided, request.cmd.special(59, 55), Mux(selected, request.cmd.special(62, 58), 0.U))
  val sharedDma = request.cmd.is_shared && !request.cmd.is_image && !request.cmd.is_config && !request.cmd.is_mmio
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
      writeRejected  := sharedDma
      write          := request
      writeGroups    := groups
      writeGroupBase := base
      writeGroup     := 0.U
      writeRow       := 0.U
      writeCursor    := request.cmd.mem_addr

      readsFinished := false.B
      writeFinal    := false.B
      writeState    := writeOffer
      when(sharedDma) {
        writeFault.error   := DmaError.Bank.U
        writeFault.address := request.cmd.mem_addr
        writeState         := writeComplete
      }
    }.otherwise {
      read          := request
      readGroups    := groups
      readGroupBase := base
      readGroup     := 0.U
      readRow       := Mux(
        request.cmd.is_strided,
        request.cmd.bank_base,
        Mux(request.cmd.is_mvin_2d, request.cmd.special(58, 53), 0.U)
      )
      readBlockBase := request.cmd.bank_base
      readBlockRow  := 0.U
      readWords     := 0.U
      imageWord     := 0.U
      imageBytes    := request.cmd.iter
      imageStarted  := false.B
      readFault     := 0.U.asTypeOf(readFault)
      readRejected  := sharedDma
      readState     := Mux(request.cmd.is_config, readComplete, readOffer)
      when(sharedDma) {
        readFault.error   := DmaError.Bank.U
        readFault.address := request.cmd.mem_addr
        readState         := readComplete
      }
    }
    when(!request.cmd.is_image && !request.cmd.is_config && !request.cmd.is_mmio && !sharedDma) {
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
  val stride =
    Mux(imageRead || read.cmd.is_mvin_2d || read.cmd.is_mmio || read.cmd.is_strided, 1.U, read.cmd.special(57, 39))

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
  dma.read.bits.transfer.is_strided   := read.cmd.is_strided
  dma.read.bits.transfer.byte_stride  := read.cmd.byte_stride
  dma.read.bits.transfer.row_bytes    := read.cmd.row_bytes
  dma.read.bits.transfer.block_log2   := Mux(read.cmd.is_strided, read.cmd.special(62, 60), 0.U)
  dma.read.bits.transfer.pixel_bytes  := read.cmd.special(42, 36) << 3
  dma.read.bits.transfer.source_width := read.cmd.special(52, 43)
  dma.read.bits.transfer.core_width   := width
  when(dma.read.fire)(readState       := readReceive)
  // Bank rows take one beat per cycle; only MMIO and image beats leave the receive state.
  val readFlow   = !imageRead && !read.cmd.is_mmio
  val readDone   = readState === readComplete && readBankRows === 0.U
  val readFailed =
    dma.readResult.bits.fault.error =/= DmaError.None.U || readFault.error =/= DmaError.None.U || imageRejected
  dma.readResult.ready := readState === readReceive && (!readFlow || readFailed || io.bankWrite.write.req.ready)
  when(dma.readResult.fire) {
    readData              := dma.readResult.bits.data
    readLast              := dma.readResult.bits.last
    imageWord             := 0.U
    when(readFailed) {
      when(dma.readResult.bits.fault.error =/= DmaError.None.U)(readFault        := dma.readResult.bits.fault)
      when(imageRejected && readFault.error === DmaError.None.U)(readFault.error := DmaError.Cancelled.U)
      when(dma.readResult.bits.last)(readState                                   := Mux(imageRead, readTerminal, readComplete))
    }.elsewhen(readFlow) {
      when(dma.readResult.bits.last)(readState := readComplete)
      when(read.cmd.is_strided) {
        val blockRows = 1.U(5.W) << read.cmd.special(62, 60)
        when(readBlockRow +& 1.U === blockRows) {
          readBlockRow  := 0.U
          readBlockBase := readBlockBase + read.cmd.bank_stride
          readRow       := readBlockBase + read.cmd.bank_stride
        }.otherwise {
          readBlockRow := readBlockRow + 1.U
          readRow      := readRow + 1.U
        }
      }.elsewhen(readGroup +& 1.U === readGroups || read.cmd.is_mvin_2d) {
        readGroup := 0.U
        readRow   := readRow + 1.U
      }.otherwise(readGroup                    := readGroup + 1.U)
    }.otherwise(readState := readBank)
  }

  io.bankWrite.bank_id             := read.cmd.bank_id
  io.bankWrite.rob_id              := read.rob_id
  io.bankWrite.group_id            := readGroupBase + readGroup
  io.bankWrite.write.req.valid     := readState === readReceive && readFlow && dma.readResult.valid && !readFailed
  io.bankWrite.write.req.bits.addr := readRow
  // MVIN_2D replaces the entire row, including zero-padding unused bytes.
  io.bankWrite.write.req.bits.data :=
    Mux(
      read.cmd.is_strided,
      dma.readResult.bits.data & (~0.U(128.W) >> ((16.U - read.cmd.row_bytes) << 3)),
      Mux(
        read.cmd.is_mvin_2d && read.cmd.special(62),
        Cat(0.U(64.W), dma.readResult.bits.data(63, 0)),
        dma.readResult.bits.data
      )
    )
  for (i <- 0 until 16) io.bankWrite.write.req.bits.mask(i) := true.B
  io.bankWrite.write.resp.ready            := !read.cmd.is_mmio
  io.metadata.request.valid                := readState === readBank && read.cmd.is_mmio
  io.metadata.request.bits.addr            := read.cmd.special(55, 39) + (readRow << 4)
  io.metadata.request.bits.data            := readData
  io.metadata.request.bits.mask            := ((1.U(17.W) << read.cmd.special(63, 56)) - 1.U)(15, 0)
  io.metadata.response.ready               := readState === readBankWait && read.cmd.is_mmio
  when(io.metadata.request.fire) {
    assert(memParam.mmioEnable.B && read.cmd.special(63, 56) >= 1.U && read.cmd.special(63, 56) <= 16.U)
    assert(read.cmd.special(55, 39) +& (read.cmd.iter << 4) <= memParam.mmioTotalBytes.U)
  }
  when(io.metadata.request.fire)(readState := readBankWait)
  when(io.bankWrite.write.req.fire =/= io.bankWrite.write.resp.fire) {
    readBankRows := Mux(io.bankWrite.write.req.fire, readBankRows + 1.U, readBankRows - 1.U)
  }
  when(io.bankWrite.write.resp.fire && !io.bankWrite.write.resp.bits.ok) {
    readFault.error := DmaError.Bank.U; readFault.address := read.cmd.mem_addr
  }
  when(io.metadata.response.fire) {
    when(!io.metadata.response.bits) {
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

  val totalBytes  = write.cmd.iter * writeGroups * rowBytes.U
  val writeStride = write.cmd.special(57, 39)

  val writeBlockLog  = write.cmd.special(62, 60)
  val writeBlockRows = 1.U(17.W) << writeBlockLog
  val writeBlockMask = writeBlockRows - 1.U

  val writeContiguous = Mux(
    write.cmd.is_strided,
    write.cmd.row_bytes === 16.U && write.cmd.byte_stride === (writeBlockRows << 4),
    writeStride === 1.U
  )

  val descriptorBytes = Mux(
    writeContiguous,
    totalBytes,
    Mux(write.cmd.is_strided, write.cmd.row_bytes << writeBlockLog, writeGroups * rowBytes.U)
  )

  dma.write.valid      := writeState === writeOffer
  dma.write.bits.vaddr := writeCursor
  dma.write.bits.len   := descriptorBytes
  when(dma.write.fire) {
    writeFinal    := writeContiguous || writeRow +& Mux(write.cmd.is_strided, writeBlockRows, 1.U) === write.cmd.iter
    writeState    := writeStream
    writeDataRows := Mux(
      writeContiguous,
      write.cmd.iter * writeGroups,
      Mux(write.cmd.is_strided, writeBlockRows, writeGroups)
    )
  }
  val lastGroup = writeGroup +& 1.U === writeGroups
  val lastRow = writeRow +& 1.U === write.cmd.iter

  val last = lastGroup && Mux(
    writeContiguous,
    lastRow,
    Mux(write.cmd.is_strided, (writeRow & writeBlockMask) === writeBlockMask, true.B)
  )

  io.bankRead.inst_id                                                 := write.instId
  io.bankWrite.inst_id                                                := read.instId
  io.bankRead.bank_id                                                 := write.cmd.bank_id
  io.bankRead.rob_id                                                  := write.rob_id
  io.bankRead.group_id                                                := writeGroupBase + writeGroup
  io.bankRead.read.req.valid                                          := writeState === writeStream && !readsFinished
  io.bankRead.read.req.bits.addr                                      := Mux(
    write.cmd.is_strided,
    write.cmd.bank_base + (writeRow >> writeBlockLog) * write.cmd.bank_stride + (writeRow & writeBlockMask),
    writeRow
  )
  when(io.bankRead.read.req.fire) {
    when(lastGroup) { writeGroup := 0.U; writeRow := writeRow + 1.U }
      .otherwise(writeGroup  := writeGroup + 1.U)
    when(last)(readsFinished := true.B)
  }
  dma.writeData.valid                                                 := writeState === writeStream && io.bankRead.read.resp.valid
  dma.writeData.bits.data                                             := io.bankRead.read.resp.bits.data
  dma.writeData.bits.last                                             := writeDataRows === 1.U
  io.bankRead.read.resp.ready                                         := writeState === writeStream && dma.writeData.ready || writeState === writeDrain
  when(dma.writeData.fire)(writeDataRows                              := writeDataRows - 1.U)
  when(io.bankRead.read.req.fire =/= io.bankRead.read.resp.fire) {
    writeBankRows := Mux(io.bankRead.read.req.fire, writeBankRows + 1.U, writeBankRows - 1.U)
  }
  when(dma.writeData.fire && dma.writeData.bits.last)(writeState      := writeWait)
  dma.writeResult.ready                                               := writeState === writeWait
  when(dma.writeResult.fire) {
    when(dma.writeResult.bits.fault.error =/= DmaError.None.U) {
      writeFault := dma.writeResult.bits.fault

      writeState := writeDrain
    }.elsewhen(writeFinal)(writeState := writeComplete)
      .otherwise {
        writeCursor   := writeCursor + Mux(
          write.cmd.is_strided,
          write.cmd.byte_stride,
          writeGroups * writeStride * rowBytes.U
        )
        readsFinished := false.B
        writeState    := writeOffer
      }
  }
  when(writeState === writeDrain && writeBankRows === 0.U)(writeState := writeComplete)

  val readProducer = Mux(imageRead, 2.U, 0.U)
  for (i <- 0 until 3) {
    io.memory.control.finished(i).valid := (if (i == 1) writeState === writeComplete && !writeRejected
                                            else
                                              readDone && !readRejected && !read.cmd.is_config && readProducer === i.U)
    io.memory.control.finished(i).bits  := (if (i == 1) writeFault else readFault)
    io.memory.control.complete(i).ready := false.B
  }
  val completion = Instantiate(new Arbiter(new Response(log2Up(frontendParam.rob_entries)), 2))
  completion.io.in(0).valid                 := readDone &&
    (readRejected || read.cmd.is_config || io.memory.control.complete(readProducer).valid)
  completion.io.in(0).bits.rob_id           := read.rob_id
  completion.io.in(0).bits.fault            := Mux(
    readRejected || read.cmd.is_config,
    readFault,
    io.memory.control.complete(readProducer).bits
  )
  completion.io.in(1).valid                 := writeState === writeComplete && (writeRejected || io.memory.control.complete(1).valid)
  completion.io.in(1).bits.rob_id           := write.rob_id
  completion.io.in(1).bits.fault            := Mux(writeRejected, writeFault, io.memory.control.complete(1).bits)
  io.memory.control.complete(
    0
  ).ready                                   := readDone && !readRejected && !imageRead && !read.cmd.is_config && completion.io.in(
    0
  ).ready
  io.memory.control.complete(
    2
  ).ready                                   := readDone && imageRead && !read.cmd.is_config && completion.io.in(0).ready
  io.memory.control.complete(1).ready       := writeState === writeComplete && !writeRejected && completion.io.in(1).ready
  io.response <> completion.io.out
  when(completion.io.in(0).fire)(readState  := readIdle)
  when(completion.io.in(1).fire)(writeState := writeIdle)
  io.bankRead.is_shared                     := false.B
  io.bankWrite.is_shared                    := false.B

  // Publish the read shape before image-loader backpressure can delay DMA.
  // The setup stage accepts streams in order; coherence reservation follows that order.
  io.memory.control.read.valid := readState === readOffer
  val readAccess = dma.read.bits.access
  readAccess.context                 := read.context
  readAccess.baseVA                  := read.cmd.mem_addr
  readAccess.rows                    := Mux(
    read.cmd.is_strided,
    read.cmd.iter >> read.cmd.special(62, 60),
    Mux(imageRead, 1.U, read.cmd.iter)
  )
  readAccess.columns                 := Mux(read.cmd.is_mvin_2d, width, 1.U)
  readAccess.spanBytes               := Mux(
    read.cmd.is_strided,
    read.cmd.row_bytes << read.cmd.special(62, 60),
    Mux(imageRead, (imageBytes.pad(64) + 15.U) & ~15.U(64.W), readGroups * rowBytes.U)
  )
  readAccess.rowStride               := Mux(
    read.cmd.is_strided,
    read.cmd.byte_stride,
    Mux(
      read.cmd.is_mvin_2d,
      read.cmd.special(52, 43) * (read.cmd.special(42, 36) << 3),
      readGroups * stride * rowBytes.U
    )
  )
  readAccess.columnStride            := Mux(read.cmd.is_mvin_2d, read.cmd.special(42, 36) << 3, 0.U)
  dma.write.bits.access.context      := write.context
  dma.write.bits.access.baseVA       := write.cmd.mem_addr
  dma.write.bits.access.rows         := Mux(write.cmd.is_strided, write.cmd.iter >> writeBlockLog, write.cmd.iter)
  dma.write.bits.access.columnStride := 0.U
  dma.write.bits.access.columns      := 1.U
  dma.write.bits.access.spanBytes    := Mux(
    write.cmd.is_strided,
    write.cmd.row_bytes << writeBlockLog,
    writeGroups * rowBytes.U
  )
  dma.write.bits.access.rowStride    := Mux(
    write.cmd.is_strided,
    write.cmd.byte_stride,
    writeGroups * writeStride * rowBytes.U
  )
}
