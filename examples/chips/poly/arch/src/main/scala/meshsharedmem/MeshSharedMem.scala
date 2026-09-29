package examples.poly.meshsharedmem

import chisel3._
import chisel3.util._

class MeshSharedMem(p: MeshSharedMemParams) extends Module {

  val io = IO(new Bundle {
    val channels = Vec(p.totalChannels, new MeshChannel(p))
  })

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
    }
    if (attached.nonEmpty) {
      val arbiter            = Module(new RRArbiter(new MeshPacket(p), attached.size))
      val localResponseReady = Wire(Vec(attached.size, Bool()))
      for ((channel, port) <- attached.zipWithIndex) {
        val busy         = RegInit(false.B)
        val errorPending = RegInit(false.B)
        val errorTag     = Reg(UInt(p.tagBits.W))
        val bankValid    = io.channels(channel).request.bits.bank < p.bankCount.U
        val packet       = Wire(new MeshPacket(p))
        packet                             := 0.U.asTypeOf(new MeshPacket(p))
        packet.destRow                     := bankRows(io.channels(channel).request.bits.bank)
        packet.destCol                     := bankCols(io.channels(channel).request.bits.bank)
        packet.sourceRow                   := row.U
        packet.sourceCol                   := col.U
        packet.channel                     := channel.U
        packet.addr                        := io.channels(channel).request.bits.addr
        packet.write                       := io.channels(channel).request.bits.write
        packet.data                        := io.channels(channel).request.bits.data
        packet.mask                        := io.channels(channel).request.bits.mask
        packet.tag                         := io.channels(channel).request.bits.tag
        arbiter.io.in(port).valid          := io.channels(channel).request.valid && !busy && bankValid
        arbiter.io.in(port).bits           := packet
        io.channels(channel).request.ready := !busy && Mux(bankValid, arbiter.io.in(port).ready, true.B)
        when(io.channels(channel).request.fire) {
          busy := true.B
          when(!bankValid) {
            errorPending := true.B
            errorTag     := io.channels(channel).request.bits.tag
          }
        }

        io.channels(channel).response.valid      :=
          errorPending || (responseRouter.io.out(MeshDirection.local).valid &&
            responseRouter.io.out(MeshDirection.local).bits.channel === channel.U)
        io.channels(channel).response.bits.data  :=
          Mux(errorPending, 0.U, responseRouter.io.out(MeshDirection.local).bits.data)
        io.channels(channel).response.bits.tag   :=
          Mux(errorPending, errorTag, responseRouter.io.out(MeshDirection.local).bits.tag)
        io.channels(channel).response.bits.error :=
          errorPending || responseRouter.io.out(MeshDirection.local).bits.error
        localResponseReady(port)                 :=
          responseRouter.io.out(MeshDirection.local).bits.channel === channel.U &&
            io.channels(channel).response.ready && !errorPending
        when(responseRouter.io.out(MeshDirection.local).valid &&
          responseRouter.io.out(MeshDirection.local).bits.channel === channel.U) {
          assert(busy && !errorPending)
        }
        when(io.channels(channel).response.fire) {
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
