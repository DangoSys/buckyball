package framework.ant

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}

object ControlOp {
  val Query       = 0
  val Stage       = 1
  val Start       = 2
  val Acknowledge = 3
  val Cancel      = 4
  val Acquire     = 5
  val Release     = 6
}

class ControlRequest extends Bundle {
  val operation = UInt(7.W)
  val context   = UInt(32.W)
  val field     = UInt(32.W)
  val data      = UInt(64.W)
}

/**
 * Task registers for one software owner. Invalid descriptors and resource reuse are
 * programming errors. Execution contract violations assert at their source.
 */
@instantiable
class Control(p: Params, contexts: Int) extends Module {
  require(contexts > 0 && p.taskBits <= 64)

  @public val io = IO(new Bundle {
    val request     = Flipped(Decoupled(new ControlRequest))
    val response    = Decoupled(UInt(64.W))
    val signatures  = Input(Vec(contexts, UInt(64.W)))
    val online      = Input(Vec(contexts, Bool()))
    val storageBusy = Input(Bool())
    val start       = Vec(contexts, Decoupled(new Start(p)))
    val result      = Vec(contexts, Flipped(Decoupled(new Completion(p))))
    val cancel      = Output(Vec(contexts, Bool()))
    val inUse       = Output(Bool())
    val available   = Output(Vec(contexts, Bool()))
  })

  val group     = RegInit(false.B)
  val owned     = RegInit(VecInit(Seq.fill(contexts)(false.B)))
  val done      = RegInit(VecInit(Seq.fill(contexts)(false.B)))
  val staged    = Reg(Vec(contexts, Vec(6, UInt(64.W))))
  val fields    = RegInit(VecInit(Seq.fill(contexts)(0.U(6.W))))
  val results   = Reg(Vec(contexts, new Completion(p)))
  val pending   = RegInit(false.B)
  val answer    = Reg(UInt(64.W))
  val req       = io.request.bits
  val index     = req.context(math.max(1, log2Ceil(contexts)) - 1, 0)
  val inRange   = req.context < contexts.U
  val free      = !owned(index) && !done(index)
  val canStart  = inRange && io.online(index) && free && group &&
    fields(index) === 63.U && staged(index)(5) === io.signatures(index)
  val launching = req.operation === ControlOp.Start.U && canStart
  io.request.ready               := !pending && !reset.asBool && (!launching || io.start(index).ready)
  io.response.valid              := pending && !reset.asBool
  io.response.bits               := answer
  io.inUse                       := group
  when(io.response.fire)(pending := false.B)
  for (i <- 0 until contexts) {
    io.available(i)           := io.online(i) && !owned(i) && !done(i)
    io.start(i).valid         := io.request.valid && !pending && !reset.asBool && launching && req.context === i.U
    io.start(i).bits.task     := staged(i)(0)
    io.start(i).bits.entry    := staged(i)(1)
    io.start(i).bits.codeEnd  := staged(i)(2)
    io.start(i).bits.argument := staged(i)(3)
    io.start(i).bits.stack    := staged(i)(4)
    // Cancel accepts immediately, independently of the start-ready path.
    io.cancel(i)              := io.request.valid && !pending && !reset.asBool &&
      req.operation === ControlOp.Cancel.U && req.context === i.U && owned(i)
    io.result(i).ready        := owned(i) && !done(i) && !reset.asBool
    when(io.result(i).valid && !reset.asBool) {
      assert(owned(i) && !done(i), "Ant result arrived without a live task")
      assert(io.result(i).bits.task === staged(i)(0), "Ant completion task does not match its submission")
    }
    when(io.result(i).fire) {
      results(i) := io.result(i).bits
      owned(i)   := false.B
      done(i)    := true.B
    }
  }
  when(io.request.fire) {
    pending := true.B
    answer  := 0.U
    assert(req.operation <= ControlOp.Release.U, "Invalid Ant control operation")
    when(req.operation <= ControlOp.Cancel.U && !(req.operation === ControlOp.Query.U && req.field === 7.U)) {
      assert(inRange, "Invalid Ant context index")
    }
    switch(req.operation) {
      is(ControlOp.Query.U) {
        assert(req.field < 10.U, "Invalid Ant query field")
        when(req.field >= 8.U)(assert(done(index), "Ant result read before completion"))
        answer := MuxLookup(req.field, 0.U)(Seq(
          0.U -> io.signatures(index),
          1.U -> Cat(done(index) && results(index).cancelled, done(index), owned(index), io.online(index)),
          2.U -> p.codeBytes.U,
          3.U -> p.data.base.U,
          4.U -> p.data.bytes.U,
          5.U -> p.shared.base.U,
          6.U -> p.shared.bytes.U,
          7.U -> contexts.U,
          8.U -> results(index).task,
          9.U -> results(index).value
        ))
      }
      is(ControlOp.Stage.U) {
        assert(free, "Ant descriptor overwritten before resource release")
        assert(req.field < 6.U, "Invalid Ant descriptor field")
        when(req.field === 0.U)(assert((req.data >> p.taskBits) === 0.U, "Ant task ID overflow"))
        staged(index)(req.field(2, 0)) := req.data
        fields(index)                  := fields(index) | UIntToOH(req.field, 6)
      }
      is(ControlOp.Start.U) {
        assert(io.online(index), "Offline Ant context submitted")
        assert(free, "Ant context reused before completion acknowledgement")
        assert(group, "Ant submitted without TSS group ownership")
        assert(fields(index) === 63.U, "Incomplete Ant descriptor")
        assert(staged(index)(5) === io.signatures(index), "Ant program fingerprint mismatch")
        when(canStart) { owned(index) := true.B; fields(index) := 0.U }
      }
      is(ControlOp.Acknowledge.U) {
        assert(done(index), "Ant completion acknowledged before result")
        done(index) := false.B
      }
      is(ControlOp.Acquire.U) {
        assert(!group && !io.storageBusy, "Ant group acquired before loader completion or previous release")
        group := true.B
      }
      is(ControlOp.Release.U) {
        assert(
          group && !owned.asUInt.orR && !done.asUInt.orR && !io.storageBusy,
          "Ant group released before tasks and storage drained"
        )
        group := false.B
      }
    }
  }
}
