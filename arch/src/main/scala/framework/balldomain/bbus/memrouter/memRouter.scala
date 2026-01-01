package framework.balldomain.bbus.memrouter

import chisel3._
import chisel3.util._
import framework.balldomain.blink.{BankRead, BankWrite}
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.top.GlobalConfig
import framework.builtin.router.Router

@instantiable
class MemRouter(val b: GlobalConfig) extends Module {
  val bbusChannel = b.ballDomain.bbusChannel
  val numBalls    = b.ballDomain.ballNum

  // Calculate total input channels from all balls
  val totalReadChannels  = b.ballDomain.ballIdMappings.map(_.inBW).sum
  val totalWriteChannels = b.ballDomain.ballIdMappings.map(_.outBW).sum

  @public
  val io = IO(new Bundle {
    val bankRead_i  = Vec(numBalls, Vec(b.memDomain.bankNum, new BankRead(b)))
    val bankWrite_i = Vec(numBalls, Vec(b.memDomain.bankNum, new BankWrite(b)))

    val bankRead_o  = Vec(bbusChannel, Flipped(new BankRead(b)))
    val bankWrite_o = Vec(bbusChannel, Flipped(new BankWrite(b)))
  })

  // Build channel mappings at compile time
  // Map flat index -> (ballIdx, channelIdx)
  val readMapping: Seq[(Int, Int)] = {
    var flatIdx = 0
    val mapping = scala.collection.mutable.ArrayBuffer[(Int, Int)]()
    for (ballIdx <- 0 until numBalls) {
      val inBW = b.ballDomain.ballIdMappings(ballIdx).inBW
      for (chIdx <- 0 until inBW.min(b.memDomain.bankNum)) {
        mapping += ((ballIdx, chIdx))
        flatIdx += 1
      }
    }
    mapping.toSeq
  }

  val writeMapping: Seq[(Int, Int)] = {
    var flatIdx = 0
    val mapping = scala.collection.mutable.ArrayBuffer[(Int, Int)]()
    for (ballIdx <- 0 until numBalls) {
      val outBW = b.ballDomain.ballIdMappings(ballIdx).outBW
      for (chIdx <- 0 until outBW.min(b.memDomain.bankNum)) {
        mapping += ((ballIdx, chIdx))
        flatIdx += 1
      }
    }
    mapping.toSeq
  }

  // Create valid signals for Router inputs
  val readValidSignals  = Wire(Vec(totalReadChannels, Bool()))
  val writeValidSignals = Wire(Vec(totalWriteChannels, Bool()))

  for (i <- 0 until totalReadChannels) {
    val (ballIdx, channelIdx) = readMapping(i)
    readValidSignals(i) := io.bankRead_i(ballIdx)(channelIdx).io.req.valid
  }

  for (i <- 0 until totalWriteChannels) {
    val (ballIdx, channelIdx) = writeMapping(i)
    writeValidSignals(i) := io.bankWrite_i(ballIdx)(channelIdx).io.req.valid
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

  // Connect Router outputs to bankRead_o and bankWrite_o
  for (outIdx <- 0 until bbusChannel) {
    // Read routing - use MuxCase to select based on Router output
    val readSelectedIdx   = readRouter.io.out(outIdx).bits
    val readCanTransmit   = readRouter.io.out(outIdx).valid && io.bankRead_o(outIdx).io.req.ready
    val readReqValidCases = readMapping.zipWithIndex.map {
      case ((ballIdx, channelIdx), flatIdx) =>
        (readSelectedIdx === flatIdx.U) -> io.bankRead_i(ballIdx)(channelIdx).io.req.valid
    }
    io.bankRead_o(outIdx).io.req.valid := MuxCase(
      false.B,
      readReqValidCases :+ (!readCanTransmit) -> false.B
    )

    val readReqBitsCases = readMapping.zipWithIndex.map {
      case ((ballIdx, channelIdx), flatIdx) =>
        (readSelectedIdx === flatIdx.U) -> io.bankRead_i(ballIdx)(channelIdx).io.req.bits
    }
    io.bankRead_o(outIdx).io.req.bits := MuxCase(
      DontCare,
      readReqBitsCases :+ (!readCanTransmit) -> 0.U.asTypeOf(io.bankRead_o(outIdx).io.req.bits.cloneType)
    )

    // Build MuxCase for read rob_id and bank_id
    val readRobIdCases = readMapping.zipWithIndex.map {
      case ((ballIdx, channelIdx), flatIdx) =>
        (readSelectedIdx === flatIdx.U) -> io.bankRead_i(ballIdx)(channelIdx).rob_id
    }
    io.bankRead_o(outIdx).rob_id := MuxCase(0.U, readRobIdCases)

    val readBankIdCases = readMapping.zipWithIndex.map {
      case ((ballIdx, channelIdx), flatIdx) =>
        (readSelectedIdx === flatIdx.U) -> io.bankRead_i(ballIdx)(channelIdx).bank_id
    }
    io.bankRead_o(outIdx).bank_id := MuxCase(0.U, readBankIdCases)

    // Connect read response back
    // Only handle channels that are actually connected (in readMapping)
    for ((mapping, flatIdx) <- readMapping.zipWithIndex) {
      val (ballIdx, channelIdx) = mapping
      when(readCanTransmit && readSelectedIdx === flatIdx.U) {
        io.bankRead_i(ballIdx)(channelIdx).io.req.ready  := io.bankRead_o(outIdx).io.req.ready
        io.bankRead_i(ballIdx)(channelIdx).io.resp.valid := io.bankRead_o(outIdx).io.resp.valid
        io.bankRead_i(ballIdx)(channelIdx).io.resp.bits  := io.bankRead_o(outIdx).io.resp.bits
      }.otherwise {
        io.bankRead_i(ballIdx)(channelIdx).io.req.ready  := false.B
        io.bankRead_i(ballIdx)(channelIdx).io.resp.valid := false.B
        io.bankRead_i(ballIdx)(channelIdx).io.resp.bits  := 0.U.asTypeOf(
          io.bankRead_i(ballIdx)(channelIdx).io.resp.bits.cloneType
        )
      }
    }
    // resp.ready: bankRead_o is Flipped, so resp.ready is output (from internal), we drive it from bankRead_i
    io.bankRead_o(outIdx).io.resp.ready := MuxCase(
      false.B,
      readMapping.zipWithIndex.map {
        case ((ballIdx, channelIdx), flatIdx) =>
          (readCanTransmit && readSelectedIdx === flatIdx.U) -> io.bankRead_i(ballIdx)(channelIdx).io.resp.ready
      }
    )

    // Write routing - similar to read
    val writeSelectedIdx = writeRouter.io.out(outIdx).bits
    val writeCanTransmit = writeRouter.io.out(outIdx).valid && io.bankWrite_o(outIdx).io.req.ready

    // Build MuxCase for write request (only valid and bits, ready is input)
    val writeReqValidCases = writeMapping.zipWithIndex.map {
      case ((ballIdx, channelIdx), flatIdx) =>
        (writeSelectedIdx === flatIdx.U) -> io.bankWrite_i(ballIdx)(channelIdx).io.req.valid
    }
    io.bankWrite_o(outIdx).io.req.valid := MuxCase(
      false.B,
      writeReqValidCases :+ (!writeCanTransmit) -> false.B
    )

    val writeReqBitsCases = writeMapping.zipWithIndex.map {
      case ((ballIdx, channelIdx), flatIdx) =>
        (writeSelectedIdx === flatIdx.U) -> io.bankWrite_i(ballIdx)(channelIdx).io.req.bits
    }
    io.bankWrite_o(outIdx).io.req.bits := MuxCase(
      DontCare,
      writeReqBitsCases :+ (!writeCanTransmit) -> 0.U.asTypeOf(io.bankWrite_o(outIdx).io.req.bits.cloneType)
    )

    val writeRobIdCases = writeMapping.zipWithIndex.map {
      case ((ballIdx, channelIdx), flatIdx) =>
        (writeSelectedIdx === flatIdx.U) -> io.bankWrite_i(ballIdx)(channelIdx).rob_id
    }
    io.bankWrite_o(outIdx).rob_id := MuxCase(0.U, writeRobIdCases)

    val writeBankIdCases = writeMapping.zipWithIndex.map {
      case ((ballIdx, channelIdx), flatIdx) =>
        (writeSelectedIdx === flatIdx.U) -> io.bankWrite_i(ballIdx)(channelIdx).bank_id
    }
    io.bankWrite_o(outIdx).bank_id := MuxCase(0.U, writeBankIdCases)

    // Connect write response back
    // Only handle channels that are actually connected (in writeMapping)
    for ((mapping, flatIdx) <- writeMapping.zipWithIndex) {
      val (ballIdx, channelIdx) = mapping
      when(writeCanTransmit && writeSelectedIdx === flatIdx.U) {
        io.bankWrite_i(ballIdx)(channelIdx).io.req.ready  := io.bankWrite_o(outIdx).io.req.ready
        io.bankWrite_i(ballIdx)(channelIdx).io.resp.valid := io.bankWrite_o(outIdx).io.resp.valid
        io.bankWrite_i(ballIdx)(channelIdx).io.resp.bits  := io.bankWrite_o(outIdx).io.resp.bits
      }.otherwise {
        io.bankWrite_i(ballIdx)(channelIdx).io.req.ready  := false.B
        io.bankWrite_i(ballIdx)(channelIdx).io.resp.valid := false.B
        io.bankWrite_i(ballIdx)(channelIdx).io.resp.bits  := 0.U.asTypeOf(
          io.bankWrite_i(ballIdx)(channelIdx).io.resp.bits.cloneType
        )
      }
    }
    io.bankWrite_o(outIdx).io.resp.ready := MuxCase(
      false.B,
      writeMapping.zipWithIndex.map {
        case ((ballIdx, channelIdx), flatIdx) =>
          (writeCanTransmit && writeSelectedIdx === flatIdx.U) -> io.bankWrite_i(ballIdx)(channelIdx).io.resp.ready
      }
    )

    // Router ready signals
    readRouter.io.out(outIdx).ready  := io.bankRead_o(outIdx).io.req.ready
    writeRouter.io.out(outIdx).ready := io.bankWrite_o(outIdx).io.req.ready
  }

}
