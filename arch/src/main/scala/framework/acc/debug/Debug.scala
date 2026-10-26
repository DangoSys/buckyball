package framework.acc.debug

import sims.configs.SimParam
import framework.acc.balldomain.configs.BallDomainParam
import framework.acc.frontend.configs.FrontendParam
import framework.acc.gpdomain.configs.GpDomainParam
import framework.acc.memdomain.configs.MemDomainParam

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.acc.memdomain.PhysicalBankHash
import framework.thread.ant.Retire
import framework.acc.debug.trace.{AcceleratorEvents, InstructionEvent, Trace}

class DebugCommit(frontendParam: FrontendParam) extends Bundle {
  val robId    = UInt(log2Ceil(frontendParam.rob_entries).W)
  val instId   = UInt(64.W)
  val threadId = UInt(64.W)
  val bank     = UInt(frontendParam.bank_id_len.W)
}

/** Producer-facing debug handshake. Unlike trace, completion may wait for checked. */
class DebugPort(frontendParam: FrontendParam) extends Bundle {

  val check = Decoupled(new Bundle {
    val robId = UInt(log2Ceil(frontendParam.rob_entries).W)
    val bank  = UInt(frontendParam.bank_id_len.W)
  })

  val checked = Flipped(Decoupled(UInt(log2Ceil(frontendParam.rob_entries).W)))
  val commit  = Output(Valid(new DebugCommit(frontendParam)))
  val idle    = Output(Bool())
}

/** Frontend debug boundary: observations never share the differential-check handshake. */
class FrontendDebugPort(frontendParam: FrontendParam, simp: SimParam) extends Bundle {
  val diffTest = simp.diffTest
  val trace    = Output(Vec(4, new InstructionEvent))
  val control  = Option.when(diffTest)(new DebugPort(frontendParam))
}

/** Producer-facing thread debug connection, assembled from the owning child IP ports. */
class AcceleratorDebugPort(
  frontendParam: FrontendParam,
  memParam:      MemDomainParam,
  gpParam:       GpDomainParam,
  ballParam:     BallDomainParam,
  simp:          SimParam)
    extends Bundle {
  val diffTest = simp.diffTest

  val frontend = new FrontendDebugPort(frontendParam, simp)
  val trace    = Output(new AcceleratorEvents(frontendParam, memParam, gpParam, ballParam))

  val banks = Option.when(diffTest)(
    Output(Vec(
      memParam.bankNum + (if (memParam.sharedEnable) memParam.sharedBankNum else 0),
      new PhysicalBankHash(frontendParam)
    ))
  )

}

/** Differential bank checks may delay completion and serialize retirement. */
@instantiable
class BankCheck(
  frontendParam: FrontendParam,
  memParam:      MemDomainParam,
  simp:          SimParam,
  traceId:       Int,
  sharedOwnerId: Int)
    extends Module {
  val diffTest = simp.diffTest
  val ownerId  = frontendParam.ownerId
  require(diffTest)

  val bankCount = memParam.bankNum + (if (memParam.sharedEnable) memParam.sharedBankNum else 0)

  @public
  val io = IO(new Bundle {
    val pipeline = Flipped(new DebugPort(frontendParam))
    val banks    = Input(Vec(bankCount, new PhysicalBankHash(frontendParam)))
  })

  val collecting  = RegInit(false.B)
  val ready       = RegInit(false.B)
  val robId       = Reg(chiselTypeOf(io.pipeline.check.bits.robId))
  val bank        = Reg(chiselTypeOf(io.pipeline.check.bits.bank))
  val group       = Reg(UInt(32.W))
  val currentHash = Reg(UInt(32.W))
  val hashes      = Reg(Vec(frontendParam.rob_entries, UInt(32.W)))
  io.pipeline.check.ready := !collecting && !ready
  when(io.pipeline.check.fire) {
    robId       := io.pipeline.check.bits.robId
    bank        := io.pipeline.check.bits.bank
    group       := 0.U
    currentHash := 0.U
    collecting  := true.B
  }

  val matches = VecInit(io.banks.map(entry =>
    entry.valid && entry.threadId === ownerId.U && entry.vbankId === bank && entry.groupId === group
  ))

  val nextMatches = VecInit(io.banks.map(entry =>
    entry.valid && entry.threadId === ownerId.U && entry.vbankId === bank && entry.groupId === group + 1.U
  ))

  val physical = Mux1H(matches, io.banks)
  def rotateLeft(value: UInt, amount: Int): UInt = Cat(value(31 - amount, 0), value(31, 32 - amount))
  val mixed    = physical.statusHash ^ rotateLeft(physical.groupId, 7) ^ "h9e3779b9".U
  val nextHash = rotateLeft(currentHash, 5) ^ mixed ^ rotateLeft(mixed, 13)
  when(collecting) {
    assert(PopCount(matches) === 1.U, "Debug requires one physical bank per virtual bank group")
    when(nextMatches.asUInt.orR) {
      currentHash := nextHash
      group       := group + 1.U
    }.otherwise {
      hashes(robId) := nextHash
      collecting    := false.B
      ready         := true.B
    }
  }
  io.pipeline.checked.valid := ready
  io.pipeline.checked.bits             := robId
  when(io.pipeline.checked.fire)(ready := false.B)

  val produced = RegInit(0.U(64.W))
  when(io.pipeline.commit.valid)(produced := produced + 1.U)
  val sink: Instance[BankCheckDPI] = Instantiate(new BankCheckDPI)
  sink.io.clock         := clock
  sink.io.reset         := reset.asBool
  sink.io.instId        := io.pipeline.commit.bits.instId
  sink.io.threadId      := traceId.U
  sink.io.ownerThreadId := Mux(
    memParam.sharedEnable.B && io.pipeline.commit.bits.bank >= frontendParam.shared_bank_id_base.U,
    sharedOwnerId.U,
    traceId.U
  )
  sink.io.w0Vbank       := io.pipeline.commit.bits.bank
  sink.io.w0Hash        := hashes(io.pipeline.commit.bits.robId)
  sink.io.fire          := io.pipeline.commit.valid
  sink.io.produced      := produced
  sink.io.idle          := io.pipeline.idle
}

/** Thread-level observability: passive trace plus explicitly enabled timing-affecting checks. */
@instantiable
class Debug(
  frontendParam: FrontendParam,
  memParam:      MemDomainParam,
  gpParam:       GpDomainParam,
  ballParam:     BallDomainParam,
  simp:          SimParam,
  traceId:       Int,
  sharedOwnerId: Int)
    extends Module {
  val diffTest = simp.diffTest
  val ownerId  = frontendParam.ownerId

  @public
  val io = IO(new Bundle {
    val ant = Input(Valid(new Retire))
    val acc = Flipped(new AcceleratorDebugPort(frontendParam, memParam, gpParam, ballParam, simp))

  })

  val trace: Instance[Trace] = Instantiate(new Trace(frontendParam, memParam, gpParam, ballParam, traceId))
  trace.io.ant          := io.ant
  trace.io.acc          := io.acc.trace
  trace.io.instructions := io.acc.frontend.trace
  io.acc.frontend.control.foreach { port =>
    val check: Instance[BankCheck] =
      Instantiate(new BankCheck(frontendParam, memParam, simp, traceId, sharedOwnerId))
    check.io.pipeline <> port
    check.io.banks := io.acc.banks.get
  }
}
