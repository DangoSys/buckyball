package memcore.memory.bank

import chisel3._
import chisel3.util._
import memcore.bus.axi.Beat

case class ScratchpadParams(dataBits: Int, banks: Int, entriesPerBank: Int) {
  require(dataBits > 0 && dataBits % 8 == 0)
  require(banks >= 1 && isPow2(banks))
  require(entriesPerBank >= 2 && isPow2(entriesPerBank))
  val bytes       = dataBits / 8
  val bankBits    = math.max(1, log2Ceil(banks))
  val rowBits     = log2Ceil(entriesPerBank)
  val addressBits = math.max(1, log2Ceil(banks * entriesPerBank * bytes))
}

class ScratchpadCommand(p: ScratchpadParams) extends Bundle {
  val write = Bool()
  val addr  = UInt(p.addressBits.W)
  val beats = UInt(32.W)
}

/**
 * Explicitly managed, bank-interleaved scratchpad memory.
 *
 * This is independent storage: it has no cache tags, directory state, CHI
 * port, or implicit coherence. Contiguous stream beats stripe across banks.
 */
class RootScratchpad(p: ScratchpadParams) extends Module {

  val io = IO(new Bundle {
    val command = Flipped(Decoupled(new ScratchpadCommand(p)))
    val write   = Flipped(Decoupled(new Beat(p.dataBits)))
    val read    = Decoupled(new Beat(p.dataBits))
    val done    = Decoupled(Bool())
  })

  val memories                                  = Seq.fill(p.banks)(SyncReadMem(p.entriesPerBank, Vec(p.bytes, UInt(8.W))))
  val idle :: writing :: reading :: done :: Nil = Enum(4)
  val state                                     = RegInit(idle)
  val command                                   = Reg(new ScratchpadCommand(p))
  val issued                                    = RegInit(0.U(32.W))
  val completed                                 = RegInit(0.U(32.W))
  val pending                                   = RegInit(false.B)
  val pendingLast                               = RegInit(false.B)
  val pendingBank                               = Reg(UInt(p.bankBits.W))
  val responseValid                             = RegInit(false.B)
  val responseData                              = Reg(UInt(p.dataBits.W))
  val responseLast                              = RegInit(false.B)

  def byteAddress(beat: UInt): UInt = command.addr + beat * p.bytes.U
  def bank(address:     UInt): UInt =
    if (p.banks == 1) 0.U(0.W) else (address >> log2Ceil(p.bytes))(p.bankBits - 1, 0)
  def row(address:      UInt): UInt =
    (address >> (log2Ceil(p.bytes) + log2Ceil(p.banks)))(p.rowBits - 1, 0)

  io.command.ready := state === idle
  when(io.command.fire) {
    assert(
      io.command.bits.beats =/= 0.U && io.command.bits.addr % p.bytes.U === 0.U,
      "Scratchpad command must be nonempty and beat aligned"
    )
    command                                                := io.command.bits
    issued                                                 := 0.U
    completed                                              := 0.U
    pending                                                := false.B
    responseValid                                          := false.B
    state                                                  := Mux(io.command.bits.write, writing, reading)
  }

  io.write.ready := state === writing
  for (index <- 0 until p.banks) {
    when(io.write.fire && bank(byteAddress(issued)) === index.U) {
      memories(index).write(
        row(byteAddress(issued)),
        io.write.bits.data.asTypeOf(Vec(p.bytes, UInt(8.W))),
        io.write.bits.keep.asBools
      )
    }
  }
  when(io.write.fire) {
    val last = issued === command.beats - 1.U
    assert(io.write.bits.last === last, "Scratchpad write TLAST does not match command length")
    issued           := issued + 1.U
    when(last)(state := done)
  }

  val mayIssue     = state === reading && issued < command.beats && (!responseValid || io.read.fire)
  val issueAddress = byteAddress(issued)
  val bankReadData = Wire(Vec(p.banks, Vec(p.bytes, UInt(8.W))))
  for (index <- 0 until p.banks) {
    val data = memories(index).read(row(issueAddress), mayIssue && bank(issueAddress) === index.U)
    bankReadData(index) := data
  }
  val readData = bankReadData(pendingBank)
  io.read.valid     := responseValid
  io.read.bits.data := responseData
  io.read.bits.keep := Fill(p.bytes, 1.U(1.W))
  io.read.bits.last := responseLast
  io.read.bits.id   := 0.U
  io.read.bits.dest := 0.U
  io.read.bits.user := 0.U
  when(pending) {
    responseData  := readData.asUInt
    responseLast  := pendingLast
    responseValid := true.B
    pending       := mayIssue
  }.elsewhen(mayIssue) {
    pending := true.B
  }
  when(mayIssue) {
    pendingBank := bank(issueAddress)
    pendingLast := issued === command.beats - 1.U
    issued      := issued + 1.U
  }
  when(!pending && !mayIssue && io.read.fire) {
    responseValid := false.B
  }
  when(io.read.fire) {
    completed                := completed + 1.U
    when(responseLast)(state := done)
  }

  io.done.valid            := state === done
  io.done.bits             := false.B
  when(io.done.fire)(state := idle)
}

object EmitRootScratchpad extends App {
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new RootScratchpad(ScratchpadParams(dataBits = 256, banks = 4, entriesPerBank = 64)),
    firtoolOpts = args.drop(1) ++ Seq("--split-verilog", "-o=build"),
    args = Array("--target-dir", "build")
  )
}
