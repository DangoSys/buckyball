package memcore.memory.mesh_shm

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}

@instantiable
class MeshSharedMem(p: MeshSharedMemParams) extends Module {

  @public
  val io = IO(new Bundle {
    val channels           = Vec(p.totalChannels, new MeshChannel(p))
    val transferCommand    = Flipped(Decoupled(new MeshTransferCommand(p)))
    val transferCompletion = Decoupled(new MeshTransferCompletion(p))
    val localBanks         = Vec(p.cores.size, new MeshLocalBankPort(p.addressBits, p.localBankBits, p.dataBits, p.tagBits))
    val bankWrites         = Vec(p.bankCount, Decoupled(new MeshClientRequest(p)))
  })

  val transfer: Instance[MeshTransferController] = Instantiate(new MeshTransferController(p))
  transfer.io.command.valid    := io.transferCommand.valid
  transfer.io.command.bits     := io.transferCommand.bits
  io.transferCommand.ready     := transfer.io.command.ready
  io.transferCompletion.valid  := transfer.io.completion.valid
  io.transferCompletion.bits   := transfer.io.completion.bits
  transfer.io.completion.ready := io.transferCompletion.ready

  for (core <- p.cores.indices if p.cores(core).bankIds.isEmpty) {
    io.localBanks(core).request.valid  := false.B
    io.localBanks(core).request.bits   := 0.U.asTypeOf(io.localBanks(core).request.bits)
    io.localBanks(core).response.ready := false.B
  }

  val requests:  Seq[Seq[Instance[MeshRouter]]]   =
    Seq.tabulate(p.rows, p.cols)((row, col) => Instantiate(new MeshRouter(p, row, col)))
  val responses: Seq[Seq[Instance[MeshRouter]]]   =
    Seq.tabulate(p.rows, p.cols)((row, col) => Instantiate(new MeshRouter(p, row, col)))
  val banks:     Seq[Seq[Instance[MeshBankNode]]] =
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
    val localRequest   = requestRouter.io.out(MeshDirection.endpoint)
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
    val privateCores =
      p.cores.indices.filter(core => p.cores(core).bankIds.nonEmpty && p.coreLocations(core) == (row, col))
    val privateReady = Wire(Vec(privateCores.size, Bool()))
    val replies      = Module(new RRArbiter(new MeshPacket(p), privateCores.size + 1) {
      override lazy val lastGrant = {
        val pointer = RegInit(0.U(math.max(1, log2Ceil(privateCores.size + 1)).W))
        when(io.out.fire)(pointer := io.chosen)
        pointer
      }
    })
    replies.io.in(0) <> bank.io.response
    for ((core, index) <- privateCores.zipWithIndex) {
      val endpoint: Instance[MeshCoreEndpoint] = Instantiate(new MeshCoreEndpoint(p, core, row, col))
      endpoint.io.request.valid := localRequest.valid && localRequest.bits.privateRequest && localRequest.bits.core === core.U
      endpoint.io.request.bits  := localRequest.bits
      privateReady(index)       := localRequest.bits.core === core.U && endpoint.io.request.ready
      io.localBanks(core) <> endpoint.io.bank
      replies.io.in(index + 1) <> endpoint.io.response
    }
    localRequest.ready := Mux(
      localRequest.bits.privateRequest,
      (if (privateCores.nonEmpty) privateReady.asUInt.orR else false.B),
      bank.io.request.ready && (!localRequest.bits.write || io.bankWrites(bankIndex).ready)
    )
    when(localRequest.valid && localRequest.bits.privateRequest) {
      assert(
        privateCores.map(core => localRequest.bits.core === core.U).reduceOption(_ || _).getOrElse(false.B),
        "private request reached a node without its Core"
      )
    }
    responseRouter.io.in(MeshDirection.endpoint) <> replies.io.out

    val attached = p.channelLocations.zipWithIndex.collect {
      case ((r, c), index) if r == row && c == col => index
    } ++ (if (row == 0 && col == 0) Seq(p.totalChannels) else Seq.empty)
    if (attached.nonEmpty) {
      val arbiter            = Module(new RRArbiter(new MeshPacket(p), attached.size) {
        override lazy val lastGrant = {
          val pointer = RegInit(0.U(math.max(1, log2Ceil(attached.size)).W))
          when(io.out.fire)(pointer := io.chosen)
          pointer
        }
      })
      val localResponseReady = Wire(Vec(attached.size, Bool()))
      for ((channel, port) <- attached.zipWithIndex) {
        if (channel == p.totalChannels) {
          arbiter.io.in(port) <> transfer.io.mesh.request
          val network = responseRouter.io.out(MeshDirection.endpoint)
          transfer.io.mesh.response.valid := network.valid && network.bits.channel === channel.U
          transfer.io.mesh.response.bits  := network.bits
          localResponseReady(port)        := network.bits.channel === channel.U && transfer.io.mesh.response.ready
        } else {
          val requestPort  = io.channels(channel).request
          val responsePort = io.channels(channel).response
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
            errorPending || (responseRouter.io.out(MeshDirection.endpoint).valid &&
              responseRouter.io.out(MeshDirection.endpoint).bits.channel === channel.U)
          responseNext.bits                                    := 0.U.asTypeOf(responseNext.bits)
          responseNext.bits.data                               :=
            Mux(errorPending, 0.U, responseRouter.io.out(MeshDirection.endpoint).bits.data)
          responseNext.bits.tag                                :=
            Mux(errorPending, errorTag, responseRouter.io.out(MeshDirection.endpoint).bits.tag)
          responseNext.bits.tkeep                              := Fill(p.maskBits, 1.U(1.W))
          responseNext.bits.tlast                              := true.B
          responseNext.bits.tuser                              := Cat(
            Mux(errorPending, errorWrite, responseRouter.io.out(MeshDirection.endpoint).bits.write),
            errorPending || responseRouter.io.out(MeshDirection.endpoint).bits.error
          )
          localResponseReady(port)                             :=
            responseRouter.io.out(MeshDirection.endpoint).bits.channel === channel.U &&
              responseNext.ready && !errorPending
          when(responseRouter.io.out(MeshDirection.endpoint).valid &&
            responseRouter.io.out(MeshDirection.endpoint).bits.channel === channel.U) {
            assert(inFlight =/= 0.U && tags(responseRouter.io.out(MeshDirection.endpoint).bits.tag))
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
      requestRouter.io.in(MeshDirection.endpoint) <> arbiter.io.out
      responseRouter.io.out(MeshDirection.endpoint).ready := localResponseReady.asUInt.orR
      when(responseRouter.io.out(MeshDirection.endpoint).valid) {
        assert(
          attached.map(channel => responseRouter.io.out(MeshDirection.endpoint).bits.channel === channel.U).reduce(
            _ || _
          )
        )
      }
    } else {
      requestRouter.io.in(MeshDirection.endpoint).valid   := false.B
      requestRouter.io.in(MeshDirection.endpoint).bits    := 0.U.asTypeOf(new MeshPacket(p))
      responseRouter.io.out(MeshDirection.endpoint).ready := false.B
      when(responseRouter.io.out(MeshDirection.endpoint).valid) {
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
