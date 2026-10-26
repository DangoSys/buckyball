package framework.acc.memdomain.bank_router

import framework.lib.queue.Queue

import framework.acc.frontend.configs.FrontendParam
import framework.acc.gpdomain.configs.GpDomainParam
import framework.acc.memdomain.configs.MemDomainParam

import framework.acc.debug.trace.MemoryEvent

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.acc.memdomain._
import framework.acc.memdomain.bank_router.blt.BankLayoutTable
import framework.acc.memdomain.prm.PhysicalBankPort
import framework.thread.tlink.BankAccess

/** Allocate a renamed binding, or commit a new binding and retire its previous one. */
class BankMapping(frontendParam: FrontendParam) extends Bundle {
  val bank     = UInt(frontendParam.bank_id_len.W)
  val vbank    = UInt(frontendParam.bank_id_len.W)
  val previous = UInt(frontendParam.bank_id_len.W)
  val threadId = UInt(64.W)
  val allocate = Bool()
  val reuse    = Bool()
  val retire   = Bool()
  val write    = Bool()
}

@instantiable
class BankRouter(
  val frontendParam: FrontendParam,
  val memParam:      MemDomainParam,
  val gpParam:       GpDomainParam,
  val diffTest:      Boolean)
    extends Module {
  val sharedPorts = memParam.sharedPorts
  val ownerId     = frontendParam.ownerId

  val gpRequests =
    if (gpParam.enable)
      gpParam.memory.memoryPorts
    else 0

  @public
  val io = IO(new Bundle {

    val traceMemory = Output(Vec(
      memParam.bankRequestPorts + gpRequests + 2 + memParam.bankNum,
      new MemoryEvent
    ))

    val requests           = Vec(memParam.bankRequestPorts, Flipped(new BankPort(frontendParam, memParam)))
    val directRequest      = Vec(gpRequests + 2, Flipped(new BankPort(frontendParam, memParam)))
    val config             = Flipped(Decoupled(new LayoutConfig(frontendParam, memParam)))
    val mapping            = Flipped(Decoupled(new BankMapping(frontendParam)))
    val queryVbank         = Input(UInt(frontendParam.bank_id_len.W))
    val queryShared        = Input(Bool())
    val queryGroups        = Output(UInt(memParam.groupCountWidth.W))
    val ballLayoutRequest  = Flipped(Decoupled(new BankLayoutRequest(frontendParam)))
    val ballLayoutResponse = Decoupled(new BankLayout(memParam))
    val gpLayoutRequest    =
      Option.when(gpParam.enable)(Flipped(Decoupled(new BankLayoutRequest(frontendParam))))
    val gpLayoutResponse   =
      Option.when(gpParam.enable)(Decoupled(new BankLayout(memParam)))
    val clearBusy          = Output(Bool())
    val physical           = Vec(memParam.bankNum, Flipped(new PhysicalBankPort(memParam)))

    val hashes =
      if (diffTest) Some(Input(Vec(memParam.bankNum, UInt(32.W))))
      else None

    val bankHashes =
      if (diffTest)
        Some(Output(Vec(memParam.bankNum, new PhysicalBankHash(frontendParam))))
      else None

    val shared = new BankAccess(frontendParam, memParam)
  })

  val blt: Instance[BankLayoutTable] = Instantiate(new BankLayoutTable(frontendParam, memParam, gpParam, diffTest))
  blt.io.physical <> io.physical
  blt.io.hashes.foreach(_ := io.hashes.get)
  io.bankHashes.foreach(_ := blt.io.bank_hashes.get)
  io.traceMemory          := blt.io.traceMemory
  io.clearBusy            := blt.io.clearBusy
  blt.io.query_vbank_id   := io.queryVbank

  val aliasCount =
    2 * (memParam.virtualBankCount + (if (gpParam.enable) 2
                                      else
                                        0)) + frontendParam.rob_entries

  // Shared layout shape belongs to this thread; physical allocation stays in SHM.
  val sharedColumns = RegInit(VecInit(Seq.fill(memParam.virtualBankCount)(0.U(
    memParam.groupCountWidth.W
  ))))

  val sharedGroups =
    RegInit(VecInit(Seq.fill(aliasCount)(0.U(memParam.groupCountWidth.W))))
  when(io.shared.config.fire) {
    val config = io.shared.config.bits
    when(config.transfer) {
      val groups = sharedColumns(config.source_bank_id) +& sharedColumns(config.vbank_id)
      sharedColumns(config.source_bank_id) := 0.U
      sharedGroups(config.source_bank_id)  := 0.U
      sharedColumns(config.vbank_id)       := groups
      sharedGroups(config.vbank_id)        := groups
    }.otherwise {
      val groups = Mux(config.alloc, config.group_id +& 1.U, 0.U)
      sharedColumns(config.vbank_id) := groups
      sharedGroups(config.vbank_id)  := groups
    }
  }
  when(io.shared.mapping.fire) {
    val change = io.shared.mapping.bits
    when(change.allocate)(sharedGroups(change.vbank)  := sharedColumns(change.bank))
    when(change.retire)(sharedGroups(change.previous) := 0.U)
  }
  val sharedQuery  = RegNext(io.queryVbank, 0.U)
  val sharedResult = RegNext(sharedGroups(sharedQuery), 0.U)
  val queryShared  = RegNext(RegNext(io.queryShared, false.B), false.B)
  io.queryGroups := Mux(queryShared, sharedResult, blt.io.query_group_count)

  val owners = RegInit(VecInit((0 until aliasCount).map(i =>
    (i % (memParam.virtualBankCount + (if (gpParam.enable) 2
                                       else 0))).U(
      frontendParam.bank_id_len.W
    )
  )))

  val committed = RegInit(VecInit((0 until memParam.virtualBankCount).map(_.U(
    frontendParam.bank_id_len.W
  ))))

  // Each client has independent lookup lanes and credits; responses hold under backpressure.
  val layoutRequests  = Seq(io.ballLayoutRequest) ++ io.gpLayoutRequest.toSeq
  val layoutResponses = Seq(io.ballLayoutResponse) ++ io.gpLayoutResponse.toSeq
  for (((request, response), client) <- layoutRequests.zip(layoutResponses).zipWithIndex) {
    val replies     = Instantiate(new Queue(new BankLayout(memParam), 4))
    val outstanding = RegInit(0.U(3.W))
    request.ready        := outstanding < 4.U
    when(request.fire =/= response.fire) {
      outstanding := Mux(request.fire, outstanding + 1.U, outstanding - 1.U)
    }
    replies.io.enq.valid := RegNext(RegNext(request.fire, false.B), false.B)
    for (operand <- 0 until 3) {
      val query = blt.io.layout(3 * client + operand)
      val bank  = request.bits.banks(operand)
      query.bank := bank
      val delayedBank    = RegNext(bank, 0.U)
      val sharedCount    = RegNext(sharedGroups(delayedBank), 0.U)
      val owner          = owners(bank)
      val shared         = memParam.sharedEnable.B &&
        owner >= frontendParam.shared_bank_id_base.U &&
        owner < memParam.virtualBankCount.U
      val returnedShared = RegNext(RegNext(shared, false.B), false.B)
      val enabled        = RegNext(RegNext(request.bits.enabled(operand), false.B), false.B)
      replies.io.enq.bits.groups(operand) := Mux(enabled, Mux(returnedShared, sharedCount, query.groups), 0.U)
      when(request.fire && request.bits.enabled(operand)) {
        assert(bank < aliasCount.U, "Layout query bank exceeds the configured namespace")
      }
    }
    assert(!replies.io.enq.valid || replies.io.enq.ready, "Layout response credit was lost")
    response <> replies.io.deq
  }

  val localBusy = WireDefault(false.B)

  val sharedMapping =
    memParam.sharedEnable.B && io.mapping.bits.bank >= frontendParam.shared_bank_id_base.U &&
      io.mapping.bits.bank < memParam.virtualBankCount.U &&
      (io.mapping.bits.allocate || io.mapping.bits.retire || io.mapping.bits.write)

  val retireAllowed = !io.mapping.bits.retire || !localBusy
  blt.io.mapping.valid    := io.mapping.valid && !sharedMapping && retireAllowed
  blt.io.mapping.bits     := io.mapping.bits
  io.shared.mapping.valid := io.mapping.valid && sharedMapping
  io.shared.mapping.bits  := io.mapping.bits
  io.mapping.ready        := Mux(sharedMapping, io.shared.mapping.ready, blt.io.mapping.ready && retireAllowed)
  when(io.mapping.fire && (io.mapping.bits.allocate || io.mapping.bits.write)) {
    owners(io.mapping.bits.vbank) := io.mapping.bits.bank
  }
  when(io.mapping.fire && io.mapping.bits.retire) {
    committed(io.mapping.bits.bank) := io.mapping.bits.vbank
  }

  blt.io.config.valid    := io.config.valid && !io.config.bits.is_shared
  blt.io.config.bits     := io.config.bits
  io.shared.config.valid := io.config.valid && io.config.bits.is_shared
  io.shared.config.bits  := io.config.bits
  io.config.ready        := Mux(io.config.bits.is_shared, io.shared.config.ready, blt.io.config.ready)

  val logical = Wire(Vec(memParam.bankRequestPorts, new BankPort(frontendParam, memParam)))
  for (i <- logical.indices) {
    logical(i) <> io.requests(i)
  }

  if (memParam.sharedEnable && memParam.bankWidth == 128) {
    // Reserve channel 0 for Mesh-initiated private-Bank access only when its
    // ordinary request/response pair has drained. This preserves the normal
    // AccPipe route and prevents one-cycle SRAM responses from being lost.
    val normal             = io.requests(0)
    val routed             = logical(0)
    val local              = io.shared.local
    val normalReadPending  = RegInit(false.B)
    val normalWritePending = RegInit(false.B)
    val localPending       = RegInit(false.B)
    localBusy := localPending
    val localIsWrite = RegInit(false.B)
    val localTag     = Reg(chiselTypeOf(local.request.bits.tag))
    val normalActive = normalReadPending || normalWritePending ||
      normal.read.req.valid || normal.write.req.valid
    val localRequest = local.request.valid && !localPending && !normalActive
    val localRead    = localRequest && !local.request.bits.write
    val localWrite   = localRequest && local.request.bits.write

    routed.bank_id             := Mux(localRequest, committed(local.request.bits.bank), normal.bank_id)
    routed.group_id            := Mux(localRequest, 0.U, normal.group_id)
    routed.is_shared           := Mux(localRequest, false.B, normal.is_shared)
    routed.thread_id           := Mux(localRequest, ownerId.U(64.W), normal.thread_id)
    routed.rob_id              := Mux(localRequest, 0.U, normal.rob_id)
    routed.inst_id             := Mux(localRequest, 0.U, normal.inst_id)
    routed.read.req.valid      := localRead || (normal.read.req.valid && !localPending)
    routed.read.req.bits.addr  := Mux(localRead, local.request.bits.addr, normal.read.req.bits.addr)
    routed.write.req.valid     := localWrite || (normal.write.req.valid && !localPending)
    routed.write.req.bits.addr := Mux(localWrite, local.request.bits.addr, normal.write.req.bits.addr)
    routed.write.req.bits.data := Mux(localWrite, local.request.bits.data, normal.write.req.bits.data)
    routed.write.req.bits.mask := Mux(localWrite, VecInit(local.request.bits.mask.asBools), normal.write.req.bits.mask)
    normal.read.req.ready      := !localPending && routed.read.req.ready
    normal.write.req.ready     := !localPending && routed.write.req.ready
    local.request.ready        := !normalActive && !localPending &&
      Mux(local.request.bits.write, routed.write.req.ready, routed.read.req.ready)

    routed.read.resp.ready    := Mux(localPending && !localIsWrite, local.response.ready, normal.read.resp.ready)
    routed.write.resp.ready   := Mux(localPending && localIsWrite, local.response.ready, normal.write.resp.ready)
    normal.read.resp.valid    := routed.read.resp.valid && !localPending
    normal.read.resp.bits     := routed.read.resp.bits
    normal.write.resp.valid   := routed.write.resp.valid && !localPending
    normal.write.resp.bits    := routed.write.resp.bits
    local.response.valid      := localPending && Mux(localIsWrite, routed.write.resp.valid, routed.read.resp.valid)
    local.response.bits.data  := routed.read.resp.bits.data
    local.response.bits.tag   := localTag
    local.response.bits.error := false.B

    when(normal.read.req.fire)(normalReadPending    := true.B)
    when(normal.read.resp.fire)(normalReadPending   := false.B)
    when(normal.write.req.fire)(normalWritePending  := true.B)
    when(normal.write.resp.fire)(normalWritePending := false.B)
    when(local.request.fire) {
      assert(local.request.bits.bank <= frontendParam.vbank_id_upper_bound.U)
      assert(local.request.bits.addr < memParam.bankEntries.U)
      assert(!local.request.bits.write || local.request.bits.mask.andR)
      localPending := true.B
      localIsWrite := local.request.bits.write
      localTag     := local.request.bits.tag
    }
    when(local.response.fire)(localPending          := false.B)
  } else {
    io.shared.local.request.ready  := false.B
    io.shared.local.response.valid := false.B
    io.shared.local.response.bits  := 0.U.asTypeOf(io.shared.local.response.bits)
  }

  val routedRequests  = logical.toSeq ++ io.directRequest.toSeq
  require(
    !memParam.sharedEnable || io.shared.requests.size >= routedRequests.size,
    "Shared-bank channels must cover Ball, GP and DMA request ports"
  )
  val privateRequests = blt.io.mem_req.toSeq ++ blt.io.kernel_req.toSeq
  for (i <- routedRequests.indices) {
    val request = routedRequests(i)
    val memory  = privateRequests(i)
    val owner   = owners(request.bank_id)
    val shared  =
      memParam.sharedEnable.B && owner >= frontendParam.shared_bank_id_base.U &&
        owner < memParam.virtualBankCount.U
    memory <> request
    memory.is_shared := false.B
    if (i < io.shared.requests.size) {
      val remote      = io.shared.requests(i)
      // Private and shared responses may have different latency. Keep one route per accepted transfer.
      val readOwners  = Instantiate(new Queue(Bool(), 8))
      val writeOwners = Instantiate(new Queue(Bool(), 8))
      remote.bank_id   := request.bank_id
      remote.group_id  := request.group_id
      remote.thread_id := request.thread_id
      remote.rob_id    := request.rob_id
      remote.inst_id   := request.inst_id
      remote.is_shared := true.B

      val readReady = Mux(shared, remote.read.req.ready, memory.read.req.ready)
      memory.read.req.valid   := request.read.req.valid && !shared && readOwners.io.enq.ready
      remote.read.req.valid   := request.read.req.valid && shared && readOwners.io.enq.ready
      remote.read.req.bits    := request.read.req.bits
      request.read.req.ready  := readReady && readOwners.io.enq.ready
      readOwners.io.enq.valid := request.read.req.valid && readReady
      readOwners.io.enq.bits  := shared
      val readShared = readOwners.io.deq.bits
      val readValid  = Mux(readShared, remote.read.resp.valid, memory.read.resp.valid)
      request.read.resp.valid := readOwners.io.deq.valid && readValid
      request.read.resp.bits  := Mux(readShared, remote.read.resp.bits, memory.read.resp.bits)
      memory.read.resp.ready  := readOwners.io.deq.valid && !readShared && request.read.resp.ready
      remote.read.resp.ready  := readOwners.io.deq.valid && readShared && request.read.resp.ready
      readOwners.io.deq.ready := readValid && request.read.resp.ready

      val writeReady = Mux(shared, remote.write.req.ready, memory.write.req.ready)
      memory.write.req.valid   := request.write.req.valid && !shared && writeOwners.io.enq.ready
      remote.write.req.valid   := request.write.req.valid && shared && writeOwners.io.enq.ready
      remote.write.req.bits    := request.write.req.bits
      request.write.req.ready  := writeReady && writeOwners.io.enq.ready
      writeOwners.io.enq.valid := request.write.req.valid && writeReady
      writeOwners.io.enq.bits  := shared
      val writeShared = writeOwners.io.deq.bits
      val writeValid  = Mux(writeShared, remote.write.resp.valid, memory.write.resp.valid)
      request.write.resp.valid := writeOwners.io.deq.valid && writeValid
      request.write.resp.bits  := Mux(writeShared, remote.write.resp.bits, memory.write.resp.bits)
      memory.write.resp.ready  := writeOwners.io.deq.valid && !writeShared && request.write.resp.ready
      remote.write.resp.ready  := writeOwners.io.deq.valid && writeShared && request.write.resp.ready
      writeOwners.io.deq.ready := writeValid && request.write.resp.ready
    } else {
      memory.read.req.valid   := request.read.req.valid && !shared
      memory.write.req.valid  := request.write.req.valid && !shared
      request.read.req.ready  := memory.read.req.ready && !shared
      request.write.req.ready := memory.write.req.ready && !shared
      when(request.read.req.valid || request.write.req.valid)(assert(!shared, "Shared request has no core port"))
    }
  }

  for (i <- routedRequests.size until io.shared.requests.size) {
    val remote = io.shared.requests(i)
    remote.bank_id          := 0.U
    remote.group_id         := 0.U
    remote.thread_id        := 0.U
    remote.rob_id           := 0.U
    remote.inst_id          := 0.U
    remote.is_shared        := true.B
    remote.read.req.valid   := false.B
    remote.read.req.bits    := 0.U.asTypeOf(remote.read.req.bits)
    remote.read.resp.ready  := false.B
    remote.write.req.valid  := false.B
    remote.write.req.bits   := 0.U.asTypeOf(remote.write.req.bits)
    remote.write.resp.ready := false.B
  }
}
