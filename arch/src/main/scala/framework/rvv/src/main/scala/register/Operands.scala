package framework.rvv

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import framework.top.GlobalConfig

@instantiable
class Operands(val b: GlobalConfig) extends Module {
  val p                = b.rvv
  val wordsPerRegister = p.wordsPerRegister
  val bankCount        = p.registerBanks
  val bankBits         = log2Ceil(bankCount)
  val addressBits      = log2Ceil(32 * wordsPerRegister)
  val count            = math.max(p.laneNumber, p.memoryPorts)
  val snapshotWords    = 8 * wordsPerRegister
  val snapshotBits     = log2Ceil(snapshotWords)

  @public val io = IO(new Bundle {
    val initialize = Input(Bool())
    val busy       = Output(Bool())

    val read = Flipped(Decoupled(Vec(
      count,
      new Bundle {
        val address  = UInt(addressBits.W)
        val snapshot = Bool()
      }
    )))

    val readResult = Decoupled(Vec(count, UInt(p.wordBits.W)))

    val write = Flipped(Decoupled(Vec(
      p.laneNumber,
      new Bundle {
        val valid   = Bool()
        val address = UInt(addressBits.W)
        val data    = UInt(p.wordBits.W)
        val mask    = UInt(p.wordBits.W)
      }
    )))

    val writeDone    = Decoupled(Bool())
    val snapshot     = Flipped(Decoupled(UInt(5.W)))
    val snapshotDone = Decoupled(Bool())
  })

  val vrf = Instantiate(new VRF(b))
  vrf.io.initialize := io.initialize
  val idle :: reading :: readDone :: writing :: writeDone :: copying :: copyDone :: Nil = Enum(7)
  val state                                                                             = RegInit(idle)
  val readFrame                                                                         = Reg(chiselTypeOf(io.read.bits))
  val writeFrame                                                                        = Reg(chiselTypeOf(io.write.bits))
  val readComplete                                                                      = Reg(Vec(count, Bool()))
  val writeComplete                                                                     = Reg(Vec(p.laneNumber, Bool()))
  val results                                                                           = Reg(Vec(count, UInt(p.wordBits.W)))
  val pending                                                                           = RegInit(VecInit(Seq.fill(bankCount)(false.B)))
  val readGroups                                                                        = Reg(Vec(bankCount, Vec(count, Bool())))
  val writeGroups                                                                       = Reg(Vec(bankCount, Vec(p.laneNumber, Bool())))

  val scratch        = SyncReadMem(snapshotWords, UInt(p.wordBits.W))
  val scratchEnable  = WireDefault(false.B)
  val scratchWrite   = WireDefault(false.B)
  val scratchAddress = WireDefault(0.U(snapshotBits.W))
  val scratchData    = WireDefault(0.U(p.wordBits.W))
  val scratchRead    = scratch.readWrite(scratchAddress, scratchData, scratchEnable && !io.initialize, scratchWrite)
  val scratchPending = RegInit(false.B)
  val scratchGroup   = Reg(Vec(count, Bool()))
  val snapshotValid  = RegInit(false.B)
  val source         = Reg(UInt(5.W))
  val copyIndex      = Reg(UInt(snapshotBits.W))
  val copyPending    = RegInit(false.B)

  val available = state === idle && !vrf.io.busy && !io.initialize
  io.read.ready         := available
  io.write.ready        := available
  io.snapshot.ready     := available
  io.readResult.valid   := state === readDone && !io.initialize
  io.readResult.bits    := results
  io.writeDone.valid    := state === writeDone && !io.initialize
  io.writeDone.bits     := true.B
  io.snapshotDone.valid := state === copyDone && !io.initialize
  io.snapshotDone.bits  := true.B
  io.busy               := state =/= idle || vrf.io.busy || io.initialize

  for (bank <- 0 until bankCount) {
    val request  = vrf.io.request(bank)
    val response = vrf.io.response(bank)
    request.valid        := false.B
    request.bits.address := 0.U
    request.bits.write   := false.B
    request.bits.data    := 0.U
    request.bits.mask    := 0.U
    response.ready       := false.B

    val readCandidates  = VecInit((0 until count).map(i =>
      !readComplete(i) && !readFrame(i).snapshot && (readFrame(i).address % bankCount.U) === bank.U
    ))
    val readLane        = PriorityEncoder(readCandidates)
    val readAddress     = readFrame(readLane).address
    val readSelected    = VecInit((0 until count).map(i => readCandidates(i) && readFrame(i).address === readAddress))
    val writeCandidates =
      VecInit((0 until p.laneNumber).map(i => !writeComplete(i) && (writeFrame(i).address % bankCount.U) === bank.U))
    val writeLane       = PriorityEncoder(writeCandidates)
    val writeAddress    = writeFrame(writeLane).address
    val writeSelected   =
      VecInit((0 until p.laneNumber).map(i => writeCandidates(i) && writeFrame(i).address === writeAddress))

    when(state === reading && !io.initialize) {
      request.valid        := readCandidates.asUInt.orR && !pending(bank)
      request.bits.address := readAddress
      response.ready       := pending(bank)
      when(request.fire) {
        pending(bank)    := true.B
        readGroups(bank) := readSelected
      }
      when(response.fire) {
        pending(bank) := false.B
        for (i <- 0 until count) {
          when(readGroups(bank)(i)) {
            results(i)      := response.bits
            readComplete(i) := true.B
          }
        }
      }
    }
    when(state === writing && !io.initialize) {
      request.valid        := writeCandidates.asUInt.orR && !pending(bank)
      request.bits.address := writeAddress
      request.bits.write   := true.B
      request.bits.mask    := (0 until p.laneNumber).map(i => Mux(writeSelected(i), writeFrame(i).mask, 0.U)).reduce(_ | _)
      request.bits.data    := (0 until p.laneNumber).map(i =>
        Mux(writeSelected(i), writeFrame(i).data & writeFrame(i).mask, 0.U)
      ).reduce(_ | _)
      response.ready       := pending(bank)
      when(request.fire) {
        for {
          i <- 0 until p.laneNumber
          j <- i + 1 until p.laneNumber
        } {
          assert(!(writeSelected(i) && writeSelected(j)) || !(writeFrame(i).mask & writeFrame(j).mask).orR)
        }
        pending(bank)     := true.B
        writeGroups(bank) := writeSelected
      }
      when(response.fire) {
        pending(bank) := false.B
        for (i <- 0 until p.laneNumber) {
          when(writeGroups(bank)(i))(writeComplete(i) := true.B)
        }
      }
    }

    val copyAddress = (source * wordsPerRegister.U + copyIndex)(addressBits - 1, 0)
    when(state === copying && (copyAddress % bankCount.U) === bank.U && !io.initialize) {
      request.valid                  := !copyPending
      request.bits.address           := copyAddress
      response.ready                 := copyPending
      when(request.fire)(copyPending := true.B)
      when(response.fire) {
        scratchEnable         := true.B
        scratchWrite          := true.B
        scratchAddress        := copyIndex
        scratchData           := response.bits
        copyPending           := false.B
        when(copyIndex === (snapshotWords - 1).U) {
          snapshotValid := true.B
          state         := copyDone
        }.otherwise(copyIndex := copyIndex + 1.U)
      }
    }
  }

  val scratchCandidates = VecInit((0 until count).map(i => !readComplete(i) && readFrame(i).snapshot))
  val scratchLane       = PriorityEncoder(scratchCandidates)
  val scratchWord       = readFrame(scratchLane).address(snapshotBits - 1, 0)
  when(state === reading && !io.initialize) {
    when(scratchCandidates.asUInt.orR && !scratchPending) {
      scratchEnable  := true.B
      scratchAddress := scratchWord
      scratchPending := true.B
      for (i <- 0 until count) {
        scratchGroup(i) := scratchCandidates(i) && readFrame(i).address(snapshotBits - 1, 0) === scratchWord
      }
    }
    when(scratchPending) {
      scratchPending := false.B
      for (i <- 0 until count) {
        when(scratchGroup(i)) {
          results(i)      := scratchRead
          readComplete(i) := true.B
        }
      }
    }
    when(readComplete.asUInt.andR)(state := readDone)
  }
  when(state === writing && writeComplete.asUInt.andR)(state := writeDone)

  when(available) {
    assert(PopCount(VecInit(Seq(io.read.valid, io.write.valid, io.snapshot.valid))) <= 1.U)
  }
  when(io.read.fire) {
    assert(snapshotValid || !io.read.bits.map(_.snapshot).reduce(_ || _))
    readFrame    := io.read.bits
    readComplete := VecInit(Seq.fill(count)(false.B))
    state        := reading
  }
  when(io.write.fire) {
    writeFrame    := io.write.bits
    writeComplete := VecInit(io.write.bits.map(w => !w.valid))
    state         := writing
  }
  when(io.snapshot.fire) {
    source        := io.snapshot.bits
    copyIndex     := 0.U
    snapshotValid := false.B
    state         := copying
  }
  when(io.readResult.fire || io.writeDone.fire || io.snapshotDone.fire)(state := idle)
  when(io.initialize) {
    state          := idle
    pending        := VecInit(Seq.fill(bankCount)(false.B))
    copyPending    := false.B
    scratchPending := false.B
    snapshotValid  := false.B
  }
}
