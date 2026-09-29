package examples.poly.meshsharedmem

import chisel3._
import chisel3.util._

class MeshSharedMem(p: MeshSharedMemParams) extends Module {

  val io = IO(new Bundle {
    val channels           = Vec(p.totalChannels, new MeshChannel(p))
    val transferCommand    = Flipped(Decoupled(new MeshTransferCommand(p)))
    val transferCompletion = Decoupled(new MeshTransferCompletion(p))
    val localBanks         = Vec(p.cores.size, new MeshLocalBankPort(p))
  })

  val transfer = Module(new MeshTransferController(p))
  transfer.io.command.valid    := io.transferCommand.valid
  transfer.io.command.bits     := io.transferCommand.bits
  io.transferCommand.ready     := transfer.io.command.ready
  io.transferCompletion.valid  := transfer.io.completion.valid
  io.transferCompletion.bits   := transfer.io.completion.bits
  transfer.io.completion.ready := io.transferCompletion.ready
  for (index <- p.cores.indices) {
    io.localBanks(index).request.valid           := transfer.io.localBanks(index).request.valid
    io.localBanks(index).request.bits            := transfer.io.localBanks(index).request.bits
    transfer.io.localBanks(index).request.ready  := io.localBanks(index).request.ready
    transfer.io.localBanks(index).response.valid := io.localBanks(index).response.valid
    transfer.io.localBanks(index).response.bits  := io.localBanks(index).response.bits
    io.localBanks(index).response.ready          := transfer.io.localBanks(index).response.ready
  }

  val requests  = Seq.tabulate(p.rows, p.cols)((row, col) => Module(new MeshRouter(p, row, col)))
  val responses = Seq.tabulate(p.rows, p.cols)((row, col) => Module(new MeshRouter(p, row, col)))
  val banks     = Seq.tabulate(p.rows, p.cols)((row, col) => Module(new MeshBankNode(p, row, col)))
  val bankRows  = VecInit((0 until p.bankCount).map(i => (i / p.cols).U(p.rowBits.W)))
  val bankCols  = VecInit((0 until p.bankCount).map(i => (i % p.cols).U(p.colBits.W)))

  def link(source: DecoupledIO[MeshPacket], sink: DecoupledIO[MeshPacket]): Unit = {
    val fifo = Module(new Queue(new MeshPacket(p), 2))
    fifo.io.enq <> source
    sink <> fifo.io.deq
  }

  for {
    row <- 0 until p.rows
    col <- 0 until p.cols
  } {
    val requestRouter  = requests(row)(col)
    val responseRouter = responses(row)(col)
    val bank           = banks(row)(col)
    bank.io.request <> requestRouter.io.out(MeshDirection.local)
    responseRouter.io.in(MeshDirection.local) <> bank.io.response

    val attached = p.channelLocations.zipWithIndex.collect {
      case ((r, c), index) if r == row && c == col => index
    } ++ (if (row == 0 && col == 0) Seq(p.totalChannels) else Seq.empty)
    if (attached.nonEmpty) {
      val arbiter            = Module(new RRArbiter(new MeshPacket(p), attached.size))
      val localResponseReady = Wire(Vec(attached.size, Bool()))
      for ((channel, port) <- attached.zipWithIndex) {
        val requestPort  = if (channel == p.totalChannels) transfer.io.mesh.request else io.channels(channel).request
        val responsePort = if (channel == p.totalChannels) transfer.io.mesh.response else io.channels(channel).response
        val busy         = RegInit(false.B)
        val errorPending = RegInit(false.B)
        val errorTag     = Reg(UInt(p.tagBits.W))
        val bankValid    = requestPort.bits.bank < p.bankCount.U &&
          (if (channel == p.totalChannels) true.B
           else !(requestPort.bits.bank === p.stagingBank.U &&
             requestPort.bits.addr === p.stagingAddress.U))
        val packet       = Wire(new MeshPacket(p))
        packet                    := 0.U.asTypeOf(new MeshPacket(p))
        packet.destRow            := bankRows(requestPort.bits.bank)
        packet.destCol            := bankCols(requestPort.bits.bank)
        packet.sourceRow          := row.U
        packet.sourceCol          := col.U
        packet.channel            := channel.U
        packet.addr               := requestPort.bits.addr
        packet.write              := requestPort.bits.write
        packet.data               := requestPort.bits.data
        packet.mask               := requestPort.bits.mask
        packet.tag                := requestPort.bits.tag
        arbiter.io.in(port).valid := requestPort.valid && !busy && bankValid
        arbiter.io.in(port).bits  := packet
        requestPort.ready         := !busy && Mux(bankValid, arbiter.io.in(port).ready, true.B)
        when(requestPort.fire) {
          busy := true.B
          when(!bankValid) {
            errorPending := true.B
            errorTag     := requestPort.bits.tag
          }
        }

        responsePort.valid       :=
          errorPending || (responseRouter.io.out(MeshDirection.local).valid &&
            responseRouter.io.out(MeshDirection.local).bits.channel === channel.U)
        responsePort.bits.data   :=
          Mux(errorPending, 0.U, responseRouter.io.out(MeshDirection.local).bits.data)
        responsePort.bits.tag    :=
          Mux(errorPending, errorTag, responseRouter.io.out(MeshDirection.local).bits.tag)
        responsePort.bits.error  :=
          errorPending || responseRouter.io.out(MeshDirection.local).bits.error
        localResponseReady(port) :=
          responseRouter.io.out(MeshDirection.local).bits.channel === channel.U &&
            responsePort.ready && !errorPending
        when(responseRouter.io.out(MeshDirection.local).valid &&
          responseRouter.io.out(MeshDirection.local).bits.channel === channel.U) {
          assert(busy && !errorPending)
        }
        when(responsePort.fire) {
          assert(busy)
          busy         := false.B
          errorPending := false.B
        }
      }
      requestRouter.io.in(MeshDirection.local) <> arbiter.io.out
      responseRouter.io.out(MeshDirection.local).ready := localResponseReady.asUInt.orR
      when(responseRouter.io.out(MeshDirection.local).valid) {
        assert(
          attached.map(channel => responseRouter.io.out(MeshDirection.local).bits.channel === channel.U).reduce(_ || _)
        )
      }
    } else {
      requestRouter.io.in(MeshDirection.local).valid   := false.B
      requestRouter.io.in(MeshDirection.local).bits    := 0.U.asTypeOf(new MeshPacket(p))
      responseRouter.io.out(MeshDirection.local).ready := false.B
      when(responseRouter.io.out(MeshDirection.local).valid) {
        assert(false.B, "response reached a node without an attached channel")
      }
    }

    if (col + 1 < p.cols) {
      link(requestRouter.io.out(MeshDirection.east), requests(row)(col + 1).io.in(MeshDirection.west))
      link(requests(row)(col + 1).io.out(MeshDirection.west), requestRouter.io.in(MeshDirection.east))
      link(responseRouter.io.out(MeshDirection.east), responses(row)(col + 1).io.in(MeshDirection.west))
      link(responses(row)(col + 1).io.out(MeshDirection.west), responseRouter.io.in(MeshDirection.east))
    }
    if (row + 1 < p.rows) {
      link(requestRouter.io.out(MeshDirection.south), requests(row + 1)(col).io.in(MeshDirection.north))
      link(requests(row + 1)(col).io.out(MeshDirection.north), requestRouter.io.in(MeshDirection.south))
      link(responseRouter.io.out(MeshDirection.south), responses(row + 1)(col).io.in(MeshDirection.north))
      link(responses(row + 1)(col).io.out(MeshDirection.north), responseRouter.io.in(MeshDirection.south))
    }

    for (router <- Seq(requestRouter, responseRouter)) {
      if (col == 0) {
        router.io.in(MeshDirection.west).valid  := false.B
        router.io.in(MeshDirection.west).bits   := 0.U.asTypeOf(new MeshPacket(p))
        router.io.out(MeshDirection.west).ready := false.B
      }
      if (col == p.cols - 1) {
        router.io.in(MeshDirection.east).valid  := false.B
        router.io.in(MeshDirection.east).bits   := 0.U.asTypeOf(new MeshPacket(p))
        router.io.out(MeshDirection.east).ready := false.B
      }
      if (row == 0) {
        router.io.in(MeshDirection.north).valid  := false.B
        router.io.in(MeshDirection.north).bits   := 0.U.asTypeOf(new MeshPacket(p))
        router.io.out(MeshDirection.north).ready := false.B
      }
      if (row == p.rows - 1) {
        router.io.in(MeshDirection.south).valid  := false.B
        router.io.in(MeshDirection.south).bits   := 0.U.asTypeOf(new MeshPacket(p))
        router.io.out(MeshDirection.south).ready := false.B
      }
    }
  }
}

object EmitMeshSharedMem extends App {
  val target = args.headOption.getOrElse("build/meshsharedmem")
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new MeshSharedMem(MeshSharedMemParams.prototype),
    firtoolOpts = Array.empty[String],
    args = Array("--target-dir", target)
  )
}
