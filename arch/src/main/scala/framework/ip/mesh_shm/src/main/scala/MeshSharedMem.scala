package framework.ip.memory.mesh_shm

import framework.lib.arbiter.{RRArbiter}

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}

class MeshSharedMemIO(p: MeshSharedMemParams) extends Bundle {
  val channels           = Vec(p.totalChannels, new MeshChannel(p))
  val external           = Vec(p.externalChannels, new MeshChannel(p))
  val transferCommand    = Flipped(Decoupled(new MeshTransferCommand(p)))
  val transferCompletion = Decoupled(new MeshTransferCompletion(p))
  val localBanks         = Vec(p.threads.size, new MeshLocalBankPort(p.addressBits, p.localBankBits, p.dataBits, p.tagBits))
  val bankWrites         = Vec(p.bankCount, Decoupled(new MeshClientRequest(p)))
}

@instantiable
class MeshSharedMem(p: MeshSharedMemParams) extends Module {

  @public
  val io = IO(new MeshSharedMemIO(p))

  val transfer: Instance[MeshTransferController] = Instantiate(new MeshTransferController(p))
  transfer.io.command.valid    := io.transferCommand.valid
  transfer.io.command.bits     := io.transferCommand.bits
  io.transferCommand.ready     := transfer.io.command.ready
  io.transferCompletion.valid  := transfer.io.completion.valid
  io.transferCompletion.bits   := transfer.io.completion.bits
  transfer.io.completion.ready := io.transferCompletion.ready

  val ordering = Instantiate(new MeshChannelOrdering(p, p.totalChannels + p.externalChannels))
  for (channel <- 0 until p.totalChannels) {
    ordering.io.client(channel) <> io.channels(channel)
  }
  for (channel <- 0 until p.externalChannels) {
    ordering.io.client(p.totalChannels + channel) <> io.external(channel)
  }

  for (thread <- p.threads.indices if p.threads(thread).ports == 0) {
    io.localBanks(thread).request.valid  := false.B
    io.localBanks(thread).request.bits   := 0.U.asTypeOf(io.localBanks(thread).request.bits)
    io.localBanks(thread).response.ready := false.B
  }

  val requests:  Seq[Seq[Instance[MeshCreditRouter]]] =
    Seq.tabulate(p.rows, p.cols)((row, col) => Instantiate(new MeshCreditRouter(p, row, col)))
  val responses: Seq[Seq[Instance[MeshCreditRouter]]] =
    Seq.tabulate(p.rows, p.cols)((row, col) => Instantiate(new MeshCreditRouter(p, row, col)))
  val banks:     Seq[Seq[Instance[MeshBankNode]]]     =
    Seq.tabulate(p.rows, p.cols)((row, col) => Instantiate(new MeshBankNode(p, row, col)))
  val bankRows = VecInit((0 until p.bankCount).map(i => (i / p.cols).U(p.rowBits.W)))
  val bankCols = VecInit((0 until p.bankCount).map(i => (i % p.cols).U(p.colBits.W)))

  for {
    row <- 0 until p.rows
    col <- 0 until p.cols
  } {
    val requestRouter  = requests(row)(col)
    val responseRouter = responses(row)(col)
    val bank           = banks(row)(col)
    val bankIndex      = row * p.cols + col
    val requestEgress  = Instantiate(new MeshCreditToDecoupled(p))
    requestEgress.io.in <> requestRouter.io.out(MeshDirection.endpoint)
    val localRequest   = requestEgress.io.out
    io.bankWrites(
      bankIndex
    ).valid                             := localRequest.valid && !localRequest.bits.privateRequest && localRequest.bits.write && bank.io.request.ready
    io.bankWrites(bankIndex).bits.bank  := bankIndex.U
    io.bankWrites(bankIndex).bits.tuser := Cat(bank.io.request.bits.addr, bank.io.request.bits.write)
    io.bankWrites(bankIndex).bits.tlast := true.B
    io.bankWrites(bankIndex).bits.data  := bank.io.request.bits.data
    io.bankWrites(bankIndex).bits.mask  := bank.io.request.bits.mask
    io.bankWrites(bankIndex).bits.tag   := bank.io.request.bits.tag
    bank.io.request.valid               := localRequest.valid && !localRequest.bits.privateRequest &&
      (!localRequest.bits.write || io.bankWrites(bankIndex).ready)
    bank.io.request.bits                := localRequest.bits
    val privateThreads =
      p.threads.indices.filter(thread => p.threads(thread).ports > 0 && p.threadLocations(thread) == (row, col))
    val privateReady   = Wire(Vec(privateThreads.size, Bool()))
    val replies        = Instantiate(new RRArbiter(new MeshPacket(p), privateThreads.size + 1))
    replies.io.in(0) <> bank.io.response
    for ((thread, index) <- privateThreads.zipWithIndex) {
      val endpoint: Instance[MeshThreadEndpoint] = Instantiate(new MeshThreadEndpoint(p, thread, row, col))
      endpoint.io.request.valid := localRequest.valid && localRequest.bits.privateRequest && localRequest.bits.thread === thread.U
      endpoint.io.request.bits  := localRequest.bits
      privateReady(index)       := localRequest.bits.thread === thread.U && endpoint.io.request.ready
      io.localBanks(thread) <> endpoint.io.bank
      replies.io.in(index + 1) <> endpoint.io.response
    }
    localRequest.ready := Mux(
      localRequest.bits.privateRequest,
      (if (privateThreads.nonEmpty) privateReady.asUInt.orR else false.B),
      bank.io.request.ready && (!localRequest.bits.write || io.bankWrites(bankIndex).ready)
    )
    when(localRequest.valid && localRequest.bits.privateRequest) {
      assert(
        privateThreads.map(thread => localRequest.bits.thread === thread.U).reduceOption(_ || _).getOrElse(false.B),
        "private request reached a node without its Thread"
      )
    }
    val responseIngress = Instantiate(new MeshDecoupledToCredit(p))
    responseIngress.io.in <> replies.io.out
    responseRouter.io.in(MeshDirection.endpoint) <> responseIngress.io.out
    val requestIngress  = Instantiate(new MeshDecoupledToCredit(p))
    requestRouter.io.in(MeshDirection.endpoint) <> requestIngress.io.out
    val responseEgress  = Instantiate(new MeshCreditToDecoupled(p))
    responseEgress.io.in <> responseRouter.io.out(MeshDirection.endpoint)
    val response        = responseEgress.io.out

    val attached = p.channelLocations.zipWithIndex.collect {
      case ((r, c), index) if r == row && c == col => index
    } ++ (if (row == 0 && col == 0) (p.totalChannels until p.totalChannels + 1 + p.externalChannels) else Seq.empty)
    if (attached.nonEmpty) {
      val arbiter            = Instantiate(new RRArbiter(new MeshPacket(p), attached.size))
      val localResponseReady = Wire(Vec(attached.size, Bool()))
      for ((channel, port) <- attached.zipWithIndex) {
        if (channel == p.totalChannels) {
          arbiter.io.in(port) <> transfer.io.mesh.request
          val network = response
          transfer.io.mesh.response.valid := network.valid && network.bits.channel === channel.U
          transfer.io.mesh.response.bits  := network.bits
          localResponseReady(port)        := network.bits.channel === channel.U && transfer.io.mesh.response.ready
        } else {
          val client       =
            if (channel < p.totalChannels) ordering.io.ordered(channel) else ordering.io.ordered(channel - 1)
          val requestPort  = client.request
          val responsePort = client.response
          val inFlight     = RegInit(0.U(log2Ceil(p.maxInFlight + 1).W))
          val tags         = RegInit(0.U((1 << p.tagBits).W))
          val available    = inFlight < p.maxInFlight.U && !tags(requestPort.bits.tag)
          val errorPending = RegInit(false.B)
          val errorTag     = Reg(UInt(p.tagBits.W))
          val errorWrite   = Reg(Bool())
          val bankValid    = requestPort.bits.bank < p.visibleBankCount.U
          val packet       = Wire(new MeshPacket(p))
          packet                    := 0.U.asTypeOf(new MeshPacket(p))
          packet.tdest              := Cat(bankRows(requestPort.bits.bank), bankCols(requestPort.bits.bank))
          packet.tdata              := requestPort.bits.data
          packet.tkeep              := requestPort.bits.mask
          packet.tid                := requestPort.bits.tag
          packet.tlast              := true.B
          packet.tuser              := Cat(
            row.U(p.rowBits.W),
            col.U(p.colBits.W),
            channel.U(p.channelBits.W),
            requestPort.bits.addr,
            requestPort.bits.write,
            false.B
          )
          arbiter.io.in(port).valid := requestPort.valid && available && !errorPending && bankValid
          arbiter.io.in(port).bits  := packet
          requestPort.ready         := available && !errorPending && Mux(bankValid, arbiter.io.in(port).ready, true.B)
          when(requestPort.fire) {
            when(!bankValid) {
              errorPending := true.B
              errorTag     := requestPort.bits.tag
              errorWrite   := requestPort.bits.write
            }
          }

          val responseValid = RegInit(false.B)
          val responseBits  = Reg(new MeshClientResponse(p))
          val responseNext  = Wire(Decoupled(new MeshClientResponse(p)))
          responsePort.valid                                   := responseValid
          responsePort.bits                                    := responseBits
          responseNext.ready                                   := !responseValid || responsePort.ready
          when(responseNext.ready) {
            responseValid                         := responseNext.valid
            when(responseNext.valid)(responseBits := responseNext.bits)
          }
          responseNext.valid                                   :=
            errorPending || (response.valid &&
              response.bits.channel === channel.U)
          responseNext.bits                                    := 0.U.asTypeOf(responseNext.bits)
          responseNext.bits.data                               :=
            Mux(errorPending, 0.U, response.bits.data)
          responseNext.bits.tag                                :=
            Mux(errorPending, errorTag, response.bits.tag)
          responseNext.bits.tkeep                              := Fill(p.maskBits, 1.U(1.W))
          responseNext.bits.tlast                              := true.B
          responseNext.bits.tuser                              := Cat(
            Mux(errorPending, errorWrite, response.bits.write),
            errorPending || response.bits.error
          )
          localResponseReady(port)                             :=
            response.bits.channel === channel.U &&
              responseNext.ready && !errorPending
          when(response.valid &&
            response.bits.channel === channel.U) {
            assert(inFlight =/= 0.U && tags(response.bits.tag))
          }
          when(responsePort.fire) {
            assert(inFlight =/= 0.U && tags(responsePort.bits.tag))
          }
          when(responseNext.fire && errorPending)(errorPending := false.B)
          val issued    = Mux(requestPort.fire, 1.U((1 << p.tagBits).W) << requestPort.bits.tag, 0.U)
          val completed = Mux(responsePort.fire, 1.U((1 << p.tagBits).W) << responsePort.bits.tag, 0.U)
          tags := (tags | issued) & ~completed
          when(requestPort.fire =/= responsePort.fire) {
            inFlight := Mux(requestPort.fire, inFlight + 1.U, inFlight - 1.U)
          }
        }
      }
      requestIngress.io.in <> arbiter.io.out
      response.ready := localResponseReady.asUInt.orR
      when(response.valid) {
        assert(
          attached.map(channel => response.bits.channel === channel.U).reduce(
            _ || _
          )
        )
      }
    } else {
      requestIngress.io.in.valid := false.B
      requestIngress.io.in.bits  := 0.U.asTypeOf(new MeshPacket(p))
      response.ready             := false.B
      when(response.valid) {
        assert(false.B, "response reached a node without an attached channel")
      }
    }

    if (col + 1 < p.cols) {
      requests(row)(col + 1).io.in(MeshDirection.west) <> requestRouter.io.out(MeshDirection.east)
      requestRouter.io.in(MeshDirection.east) <> requests(row)(col + 1).io.out(MeshDirection.west)
      responses(row)(col + 1).io.in(MeshDirection.west) <> responseRouter.io.out(MeshDirection.east)
      responseRouter.io.in(MeshDirection.east) <> responses(row)(col + 1).io.out(MeshDirection.west)
    }
    if (row + 1 < p.rows) {
      requests(row + 1)(col).io.in(MeshDirection.north) <> requestRouter.io.out(MeshDirection.south)
      requestRouter.io.in(MeshDirection.south) <> requests(row + 1)(col).io.out(MeshDirection.north)
      responses(row + 1)(col).io.in(MeshDirection.north) <> responseRouter.io.out(MeshDirection.south)
      responseRouter.io.in(MeshDirection.south) <> responses(row + 1)(col).io.out(MeshDirection.north)
    }

    for (router <- Seq(requestRouter, responseRouter)) {
      if (col == 0) {
        router.io.in(MeshDirection.west).valid   := false.B
        router.io.in(MeshDirection.west).bits    := 0.U.asTypeOf(new MeshPacket(p))
        router.io.out(MeshDirection.west).credit := false.B
      }
      if (col == p.cols - 1) {
        router.io.in(MeshDirection.east).valid   := false.B
        router.io.in(MeshDirection.east).bits    := 0.U.asTypeOf(new MeshPacket(p))
        router.io.out(MeshDirection.east).credit := false.B
      }
      if (row == 0) {
        router.io.in(MeshDirection.north).valid   := false.B
        router.io.in(MeshDirection.north).bits    := 0.U.asTypeOf(new MeshPacket(p))
        router.io.out(MeshDirection.north).credit := false.B
      }
      if (row == p.rows - 1) {
        router.io.in(MeshDirection.south).valid   := false.B
        router.io.in(MeshDirection.south).bits    := 0.U.asTypeOf(new MeshPacket(p))
        router.io.out(MeshDirection.south).credit := false.B
      }
    }
  }
}
