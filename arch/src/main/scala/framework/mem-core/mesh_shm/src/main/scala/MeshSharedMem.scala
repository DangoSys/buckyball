package memcore.memory.mesh_shm

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{Instance, Instantiate}

class MeshSharedMem(p: MeshSharedMemParams) extends Module {

  val io = IO(new Bundle {
    val channels           = Vec(p.totalChannels, new MeshChannel(p))
    val transferCommand    = Flipped(Decoupled(new MeshTransferCommand(p)))
    val transferCompletion = Decoupled(new MeshTransferCompletion(p))
    val localBanks         = Vec(p.cores.size, new MeshLocalBankPort(p.global, p.addressBits, p.localBankBits, p.tagBits))
    val bankWrites         = Output(Vec(p.bankCount, Valid(new MeshEventBeat(p.global, p.addressBits, p.bankBits, p.tagBits))))
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
  val banks: Seq[Seq[Instance[MeshBankNode]]] =
    Seq.tabulate(p.rows, p.cols)((row, col) => Instantiate(new MeshBankNode(p, row, col)))
  val bankRows = VecInit((0 until p.bankCount).map(i => (i / p.cols).U(p.rowBits.W)))
  val bankCols = VecInit((0 until p.bankCount).map(i => (i % p.cols).U(p.colBits.W)))

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
    val bankIndex      = row * p.cols + col
    io.bankWrites(bankIndex).valid      := bank.io.request.fire && bank.io.request.bits.tuser === MeshEvent.WriteRequest
    io.bankWrites(bankIndex).bits.tdest := bankIndex.U
    io.bankWrites(bankIndex).bits.addr  := bank.io.request.bits.addr
    io.bankWrites(bankIndex).bits.tuser := bank.io.request.bits.tuser
    io.bankWrites(bankIndex).bits.tdata := bank.io.request.bits.tdata
    io.bankWrites(bankIndex).bits.tkeep := bank.io.request.bits.tkeep
    io.bankWrites(bankIndex).bits.tlast := bank.io.request.bits.tlast
    io.bankWrites(bankIndex).bits.tid   := bank.io.request.bits.tid
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
        val errorWrite   = Reg(Bool())
        val errorBank    = Reg(UInt(p.bankBits.W))
        val errorAddr    = Reg(UInt(p.addressBits.W))
        val bankValid    = requestPort.bits.tdest < p.bankCount.U &&
          (if (channel == p.totalChannels) true.B else requestPort.bits.tdest < p.visibleBankCount.U) &&
          (if (channel == p.totalChannels) true.B
           else !(requestPort.bits.tdest === p.stagingBank.U &&
             requestPort.bits.addr === p.stagingAddress.U))
        val packet       = Wire(new MeshPacket(p))
        packet                    := 0.U.asTypeOf(new MeshPacket(p))
        packet.destRow            := bankRows(requestPort.bits.tdest)
        packet.destCol            := bankCols(requestPort.bits.tdest)
        packet.sourceRow          := row.U
        packet.sourceCol          := col.U
        packet.channel            := channel.U
        packet.addr               := requestPort.bits.addr
        packet.tuser              := requestPort.bits.tuser
        packet.tdata              := requestPort.bits.tdata
        packet.tkeep              := requestPort.bits.tkeep
        packet.tlast              := requestPort.bits.tlast
        packet.tid                := requestPort.bits.tid
        packet.tdest              := requestPort.bits.tdest
        arbiter.io.in(port).valid := requestPort.valid && !busy && bankValid
        arbiter.io.in(port).bits  := packet
        requestPort.ready         := !busy && Mux(bankValid, arbiter.io.in(port).ready, true.B)
        when(requestPort.fire) {
          assert(requestPort.bits.tlast && !requestPort.bits.tuser(1) && !requestPort.bits.tuser(2))
          busy := true.B
          when(!bankValid) {
            errorPending := true.B
            errorTag     := requestPort.bits.tid
            errorWrite   := requestPort.bits.tuser(0)
            errorBank    := requestPort.bits.tdest
            errorAddr    := requestPort.bits.addr
          }
        }

        responsePort.valid       :=
          errorPending || (responseRouter.io.out(MeshDirection.local).valid &&
            responseRouter.io.out(MeshDirection.local).bits.channel === channel.U)
        responsePort.bits.tdata  :=
          Mux(errorPending, 0.U, responseRouter.io.out(MeshDirection.local).bits.tdata)
        responsePort.bits.tkeep  :=
          Mux(errorPending, 0.U, responseRouter.io.out(MeshDirection.local).bits.tkeep)
        responsePort.bits.tlast  := Mux(errorPending, true.B, responseRouter.io.out(MeshDirection.local).bits.tlast)
        responsePort.bits.tid    :=
          Mux(errorPending, errorTag, responseRouter.io.out(MeshDirection.local).bits.tid)
        responsePort.bits.tdest  := Mux(errorPending, errorBank, responseRouter.io.out(MeshDirection.local).bits.tdest)
        responsePort.bits.tuser  :=
          Mux(errorPending, Cat(true.B, true.B, errorWrite), responseRouter.io.out(MeshDirection.local).bits.tuser)
        responsePort.bits.addr   := Mux(errorPending, errorAddr, responseRouter.io.out(MeshDirection.local).bits.addr)
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
  val target = args.headOption.getOrElse("build/mesh_shm")
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new MeshSharedMem(MeshSharedMemParams.prototype),
    firtoolOpts = Array.empty[String],
    args = Array("--target-dir", target)
  )
}
