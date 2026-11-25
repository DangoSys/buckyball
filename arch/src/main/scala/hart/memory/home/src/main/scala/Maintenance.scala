package framework.hart.memory.home

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import framework.generic.arbiter.RRArbiter
import framework.generic.queue.Queue
import framework.generic.bus.chi.{Opcode, RequestFlit, ResponseFlit}
import framework.generic.memory.coherence.LineQuery
import framework.generic.memory.coherence.configs.CoherenceParams

/**
 * Each requester owns one range. Empty lines advance here; resident lines use the Home's
 * ordinary coherence transactions and acknowledge only after their writebacks complete.
 */
@instantiable
class Maintenance(p: CoherenceParams) extends Module {
  val c         = p.chi
  val n         = p.agents
  val indexBits = math.max(1, log2Ceil(n))

  @public val io = IO(new Bundle {
    val request  = Flipped(Decoupled(new RequestFlit(c)))
    val response = Decoupled(new ResponseFlit(c))
    val req      = Decoupled(new RequestFlit(c))
    val rsp      = Flipped(Decoupled(new ResponseFlit(c)))
    val query    = Flipped(new LineQuery(p))
  })

  val active          = RegInit(VecInit(Seq.fill(n)(false.B)))
  val waiting         = RegInit(VecInit(Seq.fill(n)(false.B)))
  val finished        = RegInit(VecInit(Seq.fill(n)(false.B)))
  val errors          = Reg(Vec(n, UInt(2.W)))
  val commands        = Reg(Vec(n, new RequestFlit(c)))
  val cursor          = Reg(Vec(n, UInt(6.W)))
  val next            = RegInit(0.U(indexBits.W))
  val eligible        = VecInit((0 until n).map(i => active(i) && !waiting(i) && !finished(i)))
  val later           = VecInit((0 until n).map(i => eligible(i) && i.U >= next)).asUInt
  val selected        = PriorityEncoder(Mux(later.orR, later, eligible.asUInt))
  val address         = Cat(commands(selected).addr(c.addressBits - 1, 12), cursor(selected), 0.U(6.W))
  val incomingRange   = io.request.bits.maintenanceRange.asBool
  val incoming        = (io.request.bits.srcId - 1.U)(indexBits - 1, 0)
  val pendingConflict = io.request.valid && !incomingRange && io.request.bits.addr === address

  val requests     = Instantiate(new RRArbiter(new RequestFlit(c), 2))
  val requestQueue = Instantiate(new Queue(new RequestFlit(c), 2, false, false))
  requestQueue.io.enq <> requests.io.out
  io.req <> requestQueue.io.deq
  requests.io.in(0).valid := io.request.valid && !incomingRange
  requests.io.in(0).bits  := io.request.bits
  io.request.ready        := Mux(incomingRange, !active(incoming), requests.io.in(0).ready)
  when(io.request.fire && incomingRange) {
    val r = io.request.bits
    assert(r.srcId >= 1.U && r.srcId <= n.U, "Unknown maintenance requester")
    assert(
      r.tgtId === p.homeId.U && r.size === 6.U && r.returnNid === 0.U && r.returnTxnId === 0.U &&
        r.multiReq === 0.U && r.expCompAck === 0.U && r.order === 0.U && r.pas === 0.U &&
        r.memAttr(2) && !r.memAttr(1) && r.snpAttr === 1.U && r.allowRetry === 1.U &&
        r.exclSnoopMe === 0.U && r.pCrdType === 0.U && r.stashNidValidEndian === 0.U && r.tagOp === 0.U,
      "Unsupported maintenance range attributes"
    )
    assert(
      r.opcode === Opcode.CleanShared.U || r.opcode === Opcode.CleanInvalid.U || r.opcode === Opcode.MakeInvalid.U,
      "Range request requires a maintenance opcode"
    )
    assert(r.addr(5, 0) === 0.U && r.addr(11, 6) <= r.maintenanceLastLine, "Invalid Home maintenance range")
    active(incoming)   := true.B
    waiting(incoming)  := false.B
    finished(incoming) := false.B
    errors(incoming)   := 0.U
    commands(incoming) := r
    cursor(incoming)   := r.addr(11, 6)
  }

  io.query.address                           := address
  io.query.write                             := commands(selected).opcode =/= Opcode.CleanShared.U
  requests.io.in(1).valid                    := eligible.asUInt.orR && !pendingConflict && io.query.maintain
  requests.io.in(1).bits                     := commands(selected)
  requests.io.in(1).bits.addr                := address
  requests.io.in(1).bits.maintenanceRange    := 0.U
  requests.io.in(1).bits.maintenanceLastLine := 0.U
  val skip = eligible.asUInt.orR && !pendingConflict && !io.query.maintain
  when(requests.io.in(1).fire || skip) {
    next := Mux(selected === (n - 1).U, 0.U, selected + 1.U)
    when(skip) {
      when(cursor(selected) === commands(selected).maintenanceLastLine) {
        finished(selected) := true.B
      }.otherwise {
        cursor(selected) := cursor(selected) + 1.U
      }
    }.otherwise {
      waiting(selected) := true.B
    }
  }

  val responseOwner = (io.rsp.bits.tgtId - 1.U)(indexBits - 1, 0)
  val rangeResponse = io.rsp.bits.tgtId >= 1.U && io.rsp.bits.tgtId <= n.U &&
    active(responseOwner) && waiting(responseOwner) && io.rsp.bits.txnId === commands(responseOwner).txnId
  val responses     = Instantiate(new RRArbiter(new ResponseFlit(c), 2))
  val responseQueue = Instantiate(new Queue(new ResponseFlit(c), 2, false, false))
  responseQueue.io.enq <> responses.io.out
  io.response <> responseQueue.io.deq
  responses.io.in(0).valid        := io.rsp.valid && !rangeResponse
  responses.io.in(0).bits         := io.rsp.bits
  io.rsp.ready                    := Mux(rangeResponse, true.B, responses.io.in(0).ready)
  when(io.rsp.fire && rangeResponse) {
    assert(io.rsp.bits.opcode === Opcode.Comp.U, "Maintenance requires a completion response")
    waiting(responseOwner) := false.B
    when(io.rsp.bits.respErr =/= 0.U || cursor(responseOwner) === commands(responseOwner).maintenanceLastLine) {
      errors(responseOwner)   := io.rsp.bits.respErr
      finished(responseOwner) := true.B
    }.otherwise {
      cursor(responseOwner) := cursor(responseOwner) + 1.U
    }
  }
  val completed = PriorityEncoder(finished.asUInt)
  responses.io.in(1).valid        := finished.asUInt.orR
  responses.io.in(1).bits         := 0.U.asTypeOf(new ResponseFlit(c))
  responses.io.in(1).bits.srcId   := p.homeId.U
  responses.io.in(1).bits.tgtId   := commands(completed).srcId
  responses.io.in(1).bits.txnId   := commands(completed).txnId
  responses.io.in(1).bits.opcode  := Opcode.Comp.U
  responses.io.in(1).bits.respErr := errors(completed)
  when(responses.io.in(1).fire) {
    active(completed)   := false.B
    finished(completed) := false.B
  }
}
