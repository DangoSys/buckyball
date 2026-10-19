package memcore.memory.mesh_shm

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}

@instantiable
class MeshSharedMem(p: MeshSharedMemParams) extends Module {

  @public
  val io = IO(new Bundle {
    val channels           = Vec(p.totalChannels, new MeshChannel(p))
    val external           = Vec(p.externalChannels, new MeshChannel(p))
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

  val directionalEndpoints = p.inputChannelLocations.zip(p.outputChannelLocations).zipWithIndex.flatMap {
    case ((inputLocation, outputLocation), channel) =>
      val read          = Instantiate(new MeshClientEndpoint(
        p,
        inputLocation._1,
        inputLocation._2,
        2 * channel,
        Some(false)
      ))
      val write         = Instantiate(new MeshClientEndpoint(
        p,
        outputLocation._1,
        outputLocation._2,
        2 * channel + 1,
        Some(true)
      ))
      val clientRequest = io.channels(channel).request
      val available     = (read.io.inFlight +& write.io.inFlight) < p.maxInFlight.U &&
        !read.io.tags(clientRequest.bits.tag) && !write.io.tags(clientRequest.bits.tag)
      read.io.client.request.valid  := clientRequest.valid && available && !clientRequest.bits.write
      read.io.client.request.bits   := clientRequest.bits
      write.io.client.request.valid := clientRequest.valid && available && clientRequest.bits.write
      write.io.client.request.bits  := clientRequest.bits
      clientRequest.ready           := available && Mux(
        clientRequest.bits.write,
        write.io.client.request.ready,
        read.io.client.request.ready
      )

      val responses = Module(new RRArbiter(new MeshClientResponse(p), 2))
      responses.io.in(0) <> read.io.client.response
      responses.io.in(1) <> write.io.client.response
      io.channels(channel).response <> responses.io.out
      Seq(
        (inputLocation._1, inputLocation._2, 2 * channel, read),
        (outputLocation._1, outputLocation._2, 2 * channel + 1, write)
      )
  }

  val externalEndpoints = (0 until p.externalChannels).map { index =>
    val channel  = 2 * p.totalChannels + index
    val endpoint = Instantiate(new MeshClientEndpoint(p, 0, 0, channel, None))
    endpoint.io.client.request.valid := io.external(index).request.valid
    endpoint.io.client.request.bits  := io.external(index).request.bits
    io.external(index).request.ready := endpoint.io.client.request.ready
    io.external(index).response <> endpoint.io.client.response
    (0, 0, channel, endpoint)
  }

  val clientEndpoints = directionalEndpoints ++ externalEndpoints

  for (core <- p.cores.indices if p.cores(core).localBankNode.isEmpty) {
    io.localBanks(core).request.valid  := false.B
    io.localBanks(core).request.bits   := 0.U.asTypeOf(io.localBanks(core).request.bits)
    io.localBanks(core).response.ready := false.B
  }

  val requests:  Seq[Seq[Instance[MeshCreditRouter]]] =
    Seq.tabulate(p.rows, p.cols)((row, col) => Instantiate(new MeshCreditRouter(p, row, col)))
  val responses: Seq[Seq[Instance[MeshCreditRouter]]] =
    Seq.tabulate(p.rows, p.cols)((row, col) => Instantiate(new MeshCreditRouter(p, row, col)))
  val banks:     Seq[Seq[Instance[MeshBankNode]]]     =
    Seq.tabulate(p.rows, p.cols)((row, col) => Instantiate(new MeshBankNode(p, row, col)))
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
    val privateCores    =
      p.cores.indices.filter(core => p.localBankLocations(core).contains((row, col)))
    val privateReady    = Wire(Vec(privateCores.size, Bool()))
    val replies         = Module(new RRArbiter(new MeshPacket(p), privateCores.size + 1) {
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
    val responseIngress = Instantiate(new MeshDecoupledToCredit(p))
    responseIngress.io.in <> replies.io.out
    responseRouter.io.in(MeshDirection.endpoint) <> responseIngress.io.out

    val attached         = clientEndpoints.filter { case (r, c, _, _) => r == row && c == col }
    val transferAttached = row == 0 && col == 0
    val attachmentCount  = attached.size + (if (transferAttached) 1 else 0)
    val requestIngress   = Instantiate(new MeshDecoupledToCredit(p))
    requestRouter.io.in(MeshDirection.endpoint) <> requestIngress.io.out
    val responseEgress   = Instantiate(new MeshCreditToDecoupled(p))
    responseEgress.io.in <> responseRouter.io.out(MeshDirection.endpoint)
    val network          = responseEgress.io.out
    if (attachmentCount > 0) {
      val arbiter            = Module(new RRArbiter(new MeshPacket(p), attachmentCount) {
        override lazy val lastGrant = {
          val pointer = RegInit(0.U(math.max(1, log2Ceil(attachmentCount)).W))
          when(io.out.fire)(pointer := io.chosen)
          pointer
        }
      })
      val localResponseReady = Wire(Vec(attachmentCount, Bool()))
      for (((_, _, channel, endpoint), port) <- attached.zipWithIndex) {
        arbiter.io.in(port) <> endpoint.io.mesh.request
        endpoint.io.mesh.response.valid := network.valid && network.bits.channel === channel.U
        endpoint.io.mesh.response.bits  := network.bits
        localResponseReady(port)        := network.bits.channel === channel.U && endpoint.io.mesh.response.ready
      }
      if (transferAttached) {
        val port = attached.size
        arbiter.io.in(port) <> transfer.io.mesh.request
        transfer.io.mesh.response.valid := network.valid && network.bits.channel === p.transferChannel.U
        transfer.io.mesh.response.bits  := network.bits
        localResponseReady(port)        := network.bits.channel === p.transferChannel.U && transfer.io.mesh.response.ready
      }
      requestIngress.io.in <> arbiter.io.out
      network.ready := localResponseReady.asUInt.orR
      when(network.valid) {
        assert(
          (attached.map { case (_, _, channel, _) =>
            network.bits.channel === channel.U
          } ++
            (if (transferAttached) Seq(network.bits.channel === p.transferChannel.U) else Seq.empty)).reduce(_ || _)
        )
      }
    } else {
      requestIngress.io.in.valid := false.B
      requestIngress.io.in.bits  := 0.U.asTypeOf(new MeshPacket(p))
      network.ready              := false.B
      when(network.valid) {
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
