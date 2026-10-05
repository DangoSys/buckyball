package memcore.bus.chi

import memcore.memory.queue.Queue

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}

class AcceptedData(p: Params) extends Bundle {
  val srcId = UInt(p.nodeIdBits.W)
  val txnId = UInt(p.txnIdBits.W)
}

class RequestRetryIO(p: Params, records: Int) extends Bundle {
  val reqIn        = Flipped(Decoupled(new RequestFlit(p)))
  val reqOut       = Decoupled(new RequestFlit(p))
  val rspIn        = Flipped(Decoupled(new ResponseFlit(p)))
  val rspOut       = Decoupled(new ResponseFlit(p))
  val acceptedData = Flipped(Valid(new AcceptedData(p)))
  val pending      = Output(UInt(log2Ceil(records + 1).W))
}

// Tracks request acceptance only. The endpoint still owns TxnID lifetime through
// all data/completions. Cancellation and unsolicited surplus P-Credits are not used.
@instantiable
class RequestRetry(p: Params, nodeId: Int, records: Int = 4) extends Module {
  require(nodeId >= 0 && nodeId < (1 << p.nodeIdBits))
  require(records >= 1 && records <= 256)

  @public
  val io = IO(new RequestRetryIO(p, records))

  val free :: firstAttempt :: retryWait :: reissued :: Nil = Enum(4)
  val state                                                = RegInit(VecInit(Seq.fill(records)(free)))
  val requests                                             = Reg(Vec(records, new RequestFlit(p)))
  val retrySource                                          = Reg(Vec(records, UInt(p.nodeIdBits.W)))
  val retryType                                            = Reg(Vec(records, UInt(4.W)))
  val creditValid                                          = RegInit(VecInit(Seq.fill(records)(false.B)))
  val creditSource                                         = Reg(Vec(records, UInt(p.nodeIdBits.W)))
  val creditType                                           = Reg(Vec(records, UInt(4.W)))
  def liveTxn(id: UInt) = VecInit((0 until records).map(i => state(i) =/= free && requests(i).txnId === id))
  io.pending := PopCount(state.map(_ =/= free))

  val arb       = Module(new RRArbiter(new RequestFlit(p), records + 1))
  val output    = Module(new Queue(new RequestFlit(p), 2, pipe = true))
  output.io.enq <> arb.io.out
  io.reqOut <> output.io.deq
  val initial   = arb.io.in(0)
  val freeSlots = VecInit(state.map(_ === free))
  val newIndex  = PriorityEncoder(freeSlots.asUInt)
  val duplicate = liveTxn(io.reqIn.bits.txnId).asUInt.orR
  val available = freeSlots.asUInt.orR && !duplicate
  initial.valid  := io.reqIn.valid && available
  initial.bits   := io.reqIn.bits
  io.reqIn.ready := initial.ready && available
  when(io.reqIn.valid) {
    val r            = io.reqIn.bits
    val coherentRead = r.opcode === Opcode.ReadShared.U || r.opcode === Opcode.ReadNotSharedDirty.U ||
      r.opcode === Opcode.ReadUnique.U
    val coherent     = coherentRead || r.opcode === Opcode.Evict.U ||
      r.opcode === Opcode.WriteBackFull.U || r.opcode === Opcode.CleanInvalid.U ||
      r.opcode === Opcode.CleanShared.U || r.opcode === Opcode.MakeInvalid.U
    val noSnoop      = r.opcode === Opcode.ReadNoSnp.U || r.opcode === Opcode.WriteNoSnpFull.U ||
      r.opcode === Opcode.WriteNoSnpPtl.U
    assert(coherent || noSnoop, "Unsupported CHI retry request opcode")
    assert(
      r.multiReq === 0.U && r.order === 0.U && r.size === 6.U && r.addr(5, 0) === 0.U &&
        r.pas === 0.U && r.exclSnoopMe === 0.U && r.tagOp === 0.U &&
        r.stashNidValidEndian === 0.U && r.lpid(7, 5) === 0.U,
      "Unsupported CHI retry request attributes"
    )
    assert(r.expCompAck === coherentRead.asUInt, "Invalid retry completion-ack profile")
    when(coherent) {
      assert(
        r.returnNid === 0.U && r.returnTxnId === 0.U && r.memAttr === 12.U && r.snpAttr === 1.U,
        "Invalid retry coherent request attributes"
      )
    }.otherwise {
      assert(!r.memAttr(1) && r.snpAttr === 0.U, "Invalid retry non-snoop request attributes")
    }
    assert(io.reqIn.bits.srcId === nodeId.U, "Invalid retry requester")
    assert(!duplicate, "CHI retry requester reused a live TxnID")
    assert(
      io.reqIn.bits.allowRetry === 1.U && io.reqIn.bits.pCrdType === 0.U,
      "Initial CHI request must allow retry without a protocol credit"
    )
  }
  when(initial.fire) {
    requests(newIndex) := initial.bits
    state(newIndex)    := firstAttempt
  }

  val consumed = WireInit(VecInit(Seq.fill(records)(false.B)))
  for (i <- 0 until records) {
    val matches = VecInit((0 until records).map(c =>
      creditValid(c) &&
        creditSource(c) === retrySource(i) && creditType(c) === retryType(i)
    ))
    val credit  = PriorityEncoder(matches)
    val retry   = arb.io.in(i + 1)
    retry.valid           := state(i) === retryWait && matches.asUInt.orR
    retry.bits            := requests(i)
    // Arm CHI 3.3.1 permits retaining the original TgtID on retry. Match credit
    // against RetryAck SrcID even when the interconnect remapped that target.
    retry.bits.allowRetry := 0.U
    retry.bits.pCrdType   := retryType(i)
    when(retry.fire) {
      state(i)            := reissued
      consumed(credit)    := true.B
      creditValid(credit) := false.B
    }
  }

  val retryAck = io.rspIn.bits.opcode === Opcode.RetryAck.U
  val grant    = io.rspIn.bits.opcode === Opcode.PCrdGrant.U
  val internal = retryAck || grant
  io.rspOut.valid := io.rspIn.valid && !internal
  io.rspOut.bits  := io.rspIn.bits
  io.rspIn.ready  := internal || io.rspOut.ready

  def accepted(id: UInt): Unit = {
    val matches = liveTxn(id)
    val i       = PriorityEncoder(matches.asUInt)
    // Later data beats and split completions arrive after acceptance was recorded.
    // The endpoint, not this acceptance tracker, validates the full TxnID lifetime.
    when(matches.asUInt.orR) {
      assert(state(i) =/= retryWait, "Response accepted before request reissue")
      state(i) := free
    }
  }

  when(io.rspIn.fire) {
    val r = io.rspIn.bits
    assert(r.tgtId === nodeId.U, "Response to wrong retry requester")
    when(retryAck) {
      val matches = liveTxn(r.txnId)
      val i       = PriorityEncoder(matches.asUInt)
      assert(matches.asUInt.orR && state(i) === firstAttempt && r.respErr === 0.U, "Unexpected or repeated RetryAck")
      retrySource(i) := r.srcId
      retryType(i)   := r.pCrdType
      state(i)       := retryWait
    }.elsewhen(grant) {
      val space = VecInit((0 until records).map(i => !creditValid(i) || consumed(i)))
      val slot  = PriorityEncoder(space)
      assert(space.asUInt.orR && r.respErr === 0.U, "Excess or errored protocol credit")
      creditValid(slot)  := true.B
      creditSource(slot) := r.srcId
      creditType(slot)   := r.pCrdType
    }.elsewhen(r.opcode === Opcode.DBIDResp.U || r.opcode === Opcode.Comp.U ||
      r.opcode === Opcode.CompDBIDResp.U) {
      accepted(r.txnId)
    }
  }
  // The endpoint has already validated the consumed CompData. Its SrcID may be
  // a subordinate under DMT, so it need not match the request's Home TgtID.
  when(io.acceptedData.valid)(accepted(io.acceptedData.bits.txnId))
}
