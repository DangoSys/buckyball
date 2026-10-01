package memcore.memory.bank

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import memcore.bus.axi.Beat

case class BankSetParams(
  dataBits:       Int,
  banks:          Int,
  entriesPerBank: Int,
  tagBits:        Int = 8) {
  require(dataBits > 0 && dataBits % 8 == 0 && isPow2(dataBits / 8))
  require(tagBits > 0)
  require(banks >= 1 && isPow2(banks))
  require(entriesPerBank >= 2 && isPow2(entriesPerBank))
  val bytes       = dataBits / 8
  val bankBits    = math.max(1, log2Ceil(banks))
  val rowBits     = log2Ceil(entriesPerBank)
  val addressBits = math.max(1, log2Ceil(banks * entriesPerBank * bytes))
}

class BankSetCommand(p: BankSetParams) extends Bundle {
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
@instantiable
class BankSet(p: BankSetParams) extends Module {

  @public
  val io = IO(new Bundle {
    val command = Flipped(Decoupled(new BankSetCommand(p)))
    val write   = Flipped(Decoupled(new Beat(p.dataBits)))
    val read    = Decoupled(new Beat(p.dataBits))
    val done    = Decoupled(Bool())
  })

  val banks: Seq[Instance[Bank]] = Seq.fill(p.banks)(Instantiate(new Bank(p)))
  val idle :: writing :: writeResponse :: reading :: readResponse :: done :: Nil = Enum(6)
  val state                                                                      = RegInit(idle)
  val command                                                                    = Reg(new BankSetCommand(p))
  val beat                                                                       = RegInit(0.U(32.W))
  val wordAddress                                                                = (command.addr >> log2Ceil(p.bytes)) + beat
  val bankIndex                                                                  = if (p.banks == 1) 0.U else wordAddress(log2Ceil(p.banks) - 1, 0)
  val row                                                                        = (wordAddress >> log2Ceil(p.banks))(p.rowBits - 1, 0)
  val last                                                                       = beat === command.beats - 1.U
  val requestReady                                                               = VecInit(banks.map(_.io.request.ready))(bankIndex)
  val responseValid                                                              = VecInit(banks.map(_.io.response.valid))(bankIndex)
  val responseData                                                               = VecInit(banks.map(_.io.response.bits.data))(bankIndex)

  io.command.ready := state === idle
  when(io.command.fire) {
    assert(io.command.bits.beats =/= 0.U, "BankSet command must be nonempty")
    assert(io.command.bits.addr % p.bytes.U === 0.U, "BankSet command must be beat aligned")
    assert(
      (io.command.bits.addr +& (io.command.bits.beats * p.bytes.U)) <=
        (BigInt(p.banks) * p.entriesPerBank * p.bytes).U,
      "BankSet command exceeds capacity"
    )
    command                    := io.command.bits
    beat                       := 0.U
    state                      := Mux(io.command.bits.write, writing, reading)
  }

  for ((bank, index) <- banks.zipWithIndex) {
    bank.io.request.valid      := bankIndex === index.U &&
      ((state === writing && io.write.valid) || state === reading)
    bank.io.request.bits.addr  := row
    bank.io.request.bits.write := state === writing
    bank.io.request.bits.data  := io.write.bits.data
    bank.io.request.bits.mask  := io.write.bits.keep
    bank.io.request.bits.tag   := 0.U
    bank.io.response.ready     := bankIndex === index.U &&
      (state === writeResponse || (state === readResponse && io.read.ready))
  }

  io.write.ready := state === writing && requestReady
  when(io.write.fire) {
    assert(io.write.bits.last === last, "BankSet write TLAST does not match command length")
    state := writeResponse
  }
  when(state === reading && requestReady) {
    state := readResponse
  }

  io.read.valid     := state === readResponse && responseValid
  io.read.bits.data := responseData
  io.read.bits.keep := Fill(p.bytes, 1.U(1.W))
  io.read.bits.last := last
  io.read.bits.id   := 0.U
  io.read.bits.dest := 0.U
  io.read.bits.user := 0.U

  when((state === writeResponse && responseValid) || io.read.fire) {
    beat  := beat + 1.U
    state := Mux(last, done, Mux(command.write, writing, reading))
  }

  io.done.valid := state === done
  io.done.bits  := false.B
  when(io.done.fire) {
    state := idle
  }
}
