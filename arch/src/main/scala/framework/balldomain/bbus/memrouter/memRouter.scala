package framework.balldomain.bbus.memrouter

import chisel3._
import chisel3.util._
import framework.balldomain.blink.{BankRead, BankWrite}
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.top.GlobalConfig
import framework.builtin.router.Router

@instantiable
class MemRouter(val b: GlobalConfig) extends Module {
  val bbusChannel        = b.ballDomain.bbusChannel
  val numBalls           = b.ballDomain.ballNum
  val totalReadChannels  = b.ballDomain.ballIdMappings.map(_.inBW).sum
  val totalWriteChannels = b.ballDomain.ballIdMappings.map(_.outBW).sum

  @public
  val io = IO(new Bundle {
    val bankRead_i  = Vec(totalReadChannels, new BankRead(b))
    val bankWrite_i = Vec(totalWriteChannels, new BankWrite(b))
    val bankRead_o  = Vec(bbusChannel, Flipped(new BankRead(b)))
    val bankWrite_o = Vec(bbusChannel, Flipped(new BankWrite(b)))
  })

  // No need for mapping anymore - flat index directly corresponds to ball channels

  // Create valid signals for Router inputs
  val readValidSignals  = Wire(Vec(totalReadChannels, Bool()))
  val writeValidSignals = Wire(Vec(totalWriteChannels, Bool()))

  for (i <- 0 until totalReadChannels) {
    readValidSignals(i) := io.bankRead_i(i).io.req.valid
  }

  for (i <- 0 until totalWriteChannels) {
    writeValidSignals(i) := io.bankWrite_i(i).io.req.valid
  }

  // Instantiate Routers
  val readRouter:  Instance[Router] = Instantiate(new Router(totalReadChannels, bbusChannel))
  val writeRouter: Instance[Router] = Instantiate(new Router(totalWriteChannels, bbusChannel))

  // Connect Router inputs (io.in is input, so we connect from Wire to it)
  for (i <- 0 until totalReadChannels) {
    readRouter.io.in(i) := readValidSignals(i)
  }
  for (i <- 0 until totalWriteChannels) {
    writeRouter.io.in(i) := writeValidSignals(i)
  }

  // Initialize all bankRead_i and bankWrite_i channels with defaults
  for (i <- 0 until totalReadChannels) {
    io.bankRead_i(i).io.req.ready  := false.B
    io.bankRead_i(i).io.resp.valid := false.B
    io.bankRead_i(i).io.resp.bits  := 0.U.asTypeOf(io.bankRead_i(i).io.resp.bits.cloneType)
  }
  for (i <- 0 until totalWriteChannels) {
    io.bankWrite_i(i).io.req.ready  := false.B
    io.bankWrite_i(i).io.resp.valid := false.B
    io.bankWrite_i(i).io.resp.bits  := 0.U.asTypeOf(io.bankWrite_i(i).io.resp.bits.cloneType)
  }

  // Connect Router outputs to bankRead_o and bankWrite_o
  for (outIdx <- 0 until bbusChannel) {
    // Read routing - use MuxCase to select based on Router output
    val readSelectedIdx   = readRouter.io.out(outIdx).bits
    val readCanTransmit   = readRouter.io.out(outIdx).valid && io.bankRead_o(outIdx).io.req.ready
    val readReqValidCases = (0 until totalReadChannels).map { flatIdx =>
      (readSelectedIdx === flatIdx.U) -> io.bankRead_i(flatIdx).io.req.valid
    }
    io.bankRead_o(outIdx).io.req.valid := MuxCase(
      false.B,
      readReqValidCases :+ (!readCanTransmit) -> false.B
    )

    val readReqBitsCases = (0 until totalReadChannels).map { flatIdx =>
      (readSelectedIdx === flatIdx.U) -> io.bankRead_i(flatIdx).io.req.bits
    }
    io.bankRead_o(outIdx).io.req.bits := MuxCase(
      DontCare,
      readReqBitsCases :+ (!readCanTransmit) -> 0.U.asTypeOf(io.bankRead_o(outIdx).io.req.bits.cloneType)
    )

    // Build MuxCase for read rob_id and bank_id
    val readRobIdCases = (0 until totalReadChannels).map { flatIdx =>
      (readSelectedIdx === flatIdx.U) -> io.bankRead_i(flatIdx).rob_id
    }
    io.bankRead_o(outIdx).rob_id := MuxCase(0.U, readRobIdCases)

    val readBankIdCases = (0 until totalReadChannels).map { flatIdx =>
      (readSelectedIdx === flatIdx.U) -> io.bankRead_i(flatIdx).bank_id
    }
    io.bankRead_o(outIdx).bank_id := MuxCase(0.U, readBankIdCases)

    // Connect read response back - override defaults for selected channel
    for (flatIdx <- 0 until totalReadChannels) {
      when(readCanTransmit && readSelectedIdx === flatIdx.U) {
        io.bankRead_i(flatIdx).io.req.ready  := io.bankRead_o(outIdx).io.req.ready
        io.bankRead_i(flatIdx).io.resp.valid := io.bankRead_o(outIdx).io.resp.valid
        io.bankRead_i(flatIdx).io.resp.bits  := io.bankRead_o(outIdx).io.resp.bits
      }
    }
    // resp.ready: bankRead_o is Flipped, so resp.ready is output (from internal), we drive it from bankRead_i
    io.bankRead_o(outIdx).io.resp.ready := MuxCase(
      false.B,
      (0 until totalReadChannels).map { flatIdx =>
        (readCanTransmit && readSelectedIdx === flatIdx.U) -> io.bankRead_i(flatIdx).io.resp.ready
      }
    )

    val writeSelectedIdx = writeRouter.io.out(outIdx).bits
    val writeCanTransmit = writeRouter.io.out(outIdx).valid && io.bankWrite_o(outIdx).io.req.ready

    // Build MuxCase for write request (only valid and bits, ready is input)
    val writeReqValidCases = (0 until totalWriteChannels).map { flatIdx =>
      (writeSelectedIdx === flatIdx.U) -> io.bankWrite_i(flatIdx).io.req.valid
    }
    io.bankWrite_o(outIdx).io.req.valid := MuxCase(
      false.B,
      writeReqValidCases :+ (!writeCanTransmit) -> false.B
    )

    val writeReqBitsCases = (0 until totalWriteChannels).map { flatIdx =>
      (writeSelectedIdx === flatIdx.U) -> io.bankWrite_i(flatIdx).io.req.bits
    }
    io.bankWrite_o(outIdx).io.req.bits := MuxCase(
      DontCare,
      writeReqBitsCases :+ (!writeCanTransmit) -> 0.U.asTypeOf(io.bankWrite_o(outIdx).io.req.bits.cloneType)
    )

    val writeRobIdCases = (0 until totalWriteChannels).map { flatIdx =>
      (writeSelectedIdx === flatIdx.U) -> io.bankWrite_i(flatIdx).rob_id
    }
    io.bankWrite_o(outIdx).rob_id := MuxCase(0.U, writeRobIdCases)

    val writeBankIdCases = (0 until totalWriteChannels).map { flatIdx =>
      (writeSelectedIdx === flatIdx.U) -> io.bankWrite_i(flatIdx).bank_id
    }
    io.bankWrite_o(outIdx).bank_id := MuxCase(0.U, writeBankIdCases)

    // Connect write response back - override defaults for selected channel
    for (flatIdx <- 0 until totalWriteChannels) {
      when(writeCanTransmit && writeSelectedIdx === flatIdx.U) {
        io.bankWrite_i(flatIdx).io.req.ready  := io.bankWrite_o(outIdx).io.req.ready
        io.bankWrite_i(flatIdx).io.resp.valid := io.bankWrite_o(outIdx).io.resp.valid
        io.bankWrite_i(flatIdx).io.resp.bits  := io.bankWrite_o(outIdx).io.resp.bits
      }
    }
    io.bankWrite_o(outIdx).io.resp.ready := MuxCase(
      false.B,
      (0 until totalWriteChannels).map { flatIdx =>
        (writeCanTransmit && writeSelectedIdx === flatIdx.U) -> io.bankWrite_i(flatIdx).io.resp.ready
      }
    )

    // Router ready signals
    readRouter.io.out(outIdx).ready  := io.bankRead_o(outIdx).io.req.ready
    writeRouter.io.out(outIdx).ready := io.bankWrite_o(outIdx).io.req.ready
  }

}
