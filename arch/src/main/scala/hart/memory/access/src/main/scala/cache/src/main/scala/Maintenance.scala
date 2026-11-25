package framework.hart.memory.access.cache

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.generic.bus.chi.{HomeMapping, Opcode, RequestFlit, ResponseFlit, Params => ChiParams}
import framework.generic.memory.interlock.{Acknowledgement, Maintenance => Range, MaintenanceOp, Params}

/** Serial range CMO client sharing the CPU's CHI node, with a distinct transaction ID. */
@instantiable
class Maintenance(
  c:           ChiParams,
  nodeId:      Int,
  mapping:     HomeMapping,
  transaction: Int,
  tracking:    Params,
  rangeHome:   Boolean)
    extends Module {
  require(tracking.addressBits == c.addressBits && tracking.lineBytes == 64)
  require(BigInt(transaction) < (BigInt(1) << c.txnIdBits))

  @public
  val io = IO(new Bundle {
    val request  = Flipped(Decoupled(new Range(tracking)))
    val response = Decoupled(new Acknowledgement(tracking))
    // The caller drains older accepted CPU accesses and fills before asserting this.
    val drained  = Input(Bool())

    val req = Decoupled(new RequestFlit(c))

    val rsp = Flipped(Decoupled(new ResponseFlit(c)))

    val idle = Output(Bool())
  })

  val idle :: drain :: send :: waitResponse :: respond :: Nil = Enum(5)
  val state                                                   = RegInit(idle)
  val command                                                 = Reg(new Range(tracking))
  val address                                                 = Reg(UInt(c.addressBits.W))
  val succeeded                                               = RegInit(false.B)
  io.idle                                   := state === idle
  io.request.ready                          := state === idle
  when(io.request.fire) {
    val legal = io.request.bits.firstLine(5, 0) === 0.U && io.request.bits.lastLine(5, 0) === 0.U &&
      io.request.bits.firstLine <= io.request.bits.lastLine && io.request.bits.op <= MaintenanceOp.Invalidate.U
    assert(
      io.request.bits.firstLine(5, 0) === 0.U && io.request.bits.lastLine(5, 0) === 0.U,
      "Maintenance endpoints must be cache-line aligned"
    )
    assert(io.request.bits.firstLine <= io.request.bits.lastLine, "Maintenance range is reversed")
    assert(io.request.bits.op <= MaintenanceOp.Invalidate.U, "Unknown maintenance operation")
    command   := io.request.bits
    address   := io.request.bits.firstLine
    succeeded := legal
    state     := Mux(legal, drain, respond)
  }
  when(state === drain && io.drained)(state := send)

  io.req.valid            := state === send
  io.req.bits             := 0.U.asTypeOf(new RequestFlit(c))
  io.req.bits.srcId       := nodeId.U
  io.req.bits.tgtId       := mapping.node(address)
  io.req.bits.txnId       := transaction.U
  io.req.bits.addr        := address
  io.req.bits.size        := 6.U
  io.req.bits.opcode      := MuxLookup(command.op, Opcode.MakeInvalid.U)(Seq(
    MaintenanceOp.Clean.U           -> Opcode.CleanShared.U,
    MaintenanceOp.CleanInvalidate.U -> Opcode.CleanInvalid.U,
    MaintenanceOp.Invalidate.U      -> Opcode.MakeInvalid.U
  ))
  io.req.bits.snpAttr     := 1.U
  io.req.bits.memAttr     := "b1100".U
  io.req.bits.allowRetry  := 1.U
  if (rangeHome) {
    val multipleLines = command.firstLine =/= command.lastLine
    io.req.bits.maintenanceRange    := multipleLines.asUInt
    io.req.bits.maintenanceLastLine := Mux(multipleLines, command.lastLine(11, 6), 0.U)
    when(io.request.fire) {
      assert(
        io.request.bits.firstLine(c.addressBits - 1, 12) === io.request.bits.lastLine(c.addressBits - 1, 12),
        "Home maintenance range crosses a page"
      )
    }
  }
  when(io.req.fire)(state := waitResponse)

  io.rsp.ready                 := state === waitResponse
  when(io.rsp.valid) {
    assert(state === waitResponse || io.req.fire, "Maintenance response has no issued request")
  }
  when(io.rsp.fire) {
    val r        = io.rsp.bits
    val identity = r.srcId === mapping.node(address) && r.tgtId === nodeId.U && r.txnId === transaction.U
    assert(identity, "Maintenance completion identity mismatch")
    assert(r.opcode === Opcode.Comp.U, "Maintenance requires a completion response")
    when(!identity || r.opcode =/= Opcode.Comp.U || r.respErr =/= 0.U) {
      succeeded := false.B
      state     := respond
    }.elsewhen((if (rangeHome) true.B else address === command.lastLine)) {
      state := respond
    }.otherwise {
      address := address + 64.U
      state   := send
    }
  }
  io.response.valid            := state === respond
  io.response.bits.tag         := command.tag
  io.response.bits.ok          := succeeded
  when(io.response.fire)(state := idle)
}
