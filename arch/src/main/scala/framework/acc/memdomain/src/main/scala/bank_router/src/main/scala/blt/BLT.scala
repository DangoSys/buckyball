package framework.acc.memdomain.bank_router.blt

import framework.lib.arbiter.{RRArbiter}

import framework.acc.frontend.configs.FrontendParam
import framework.acc.gpdomain.configs.GpDomainParam
import framework.acc.memdomain.configs.MemDomainParam

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.acc.memdomain.{LayoutConfig, LayoutQuery}
import framework.acc.memdomain.BankPort
import framework.acc.debug.trace.MemoryEvent
import framework.acc.memdomain.bank_router.{BankMapping, BankMappingIO}
import framework.acc.memdomain.prm.PhysicalBankPort
import framework.acc.memdomain.PhysicalBankHash

class BankRequest(memp: MemDomainParam) extends Bundle {
  val write = Bool()
  val addr  = UInt(16.W)
  val data  = UInt(memp.bankWidth.W)
  val mask  = Vec(memp.bankMaskLen, Bool())
}

@instantiable
class BankLayoutTable(
  val frontendParam: FrontendParam,
  val memParam:      MemDomainParam,
  val gpParam:       GpDomainParam,
  val diffTest:      Boolean)
    extends Module {

  val kernelRequests =
    if (gpParam.enable)
      gpParam.memory.memoryPorts
    else 0

  val requestCount = memParam.bankRequestPorts + kernelRequests + 2

  @public
  val io = IO(new Bundle {
    val traceMemory =
      Output(Vec(requestCount + memParam.bankNum, new MemoryEvent))
    val mem_req     = Vec(memParam.bankRequestPorts, Flipped(new BankPort(frontendParam, memParam)))
    val kernel_req  = Vec(kernelRequests + 2, Flipped(new BankPort(frontendParam, memParam)))
    val config      = Flipped(Decoupled(new LayoutConfig(frontendParam, memParam)))
    val mapping     = Flipped(new BankMappingIO(frontendParam))

    // Query interface for frontend to get group count
    val query_vbank_id    = Input(UInt(frontendParam.bank_id_len.W))
    val query_group_count = Output(UInt(memParam.groupCountWidth.W))
    val layout            =
      Vec(3 * (if (gpParam.enable) 2 else 1), Flipped(new LayoutQuery(frontendParam, memParam)))
    val clearBusy         = Output(Bool())
    val physical          = Vec(memParam.bankNum, Flipped(new PhysicalBankPort(memParam)))

    val hashes =
      if (diffTest) Some(Input(Vec(memParam.bankNum, UInt(32.W))))
      else None

    val bank_hashes =
      if (diffTest)
        Some(Output(Vec(memParam.bankNum, new PhysicalBankHash(frontendParam))))
      else None

  })

  val requests = io.mem_req.toSeq ++ io.kernel_req.toSeq

  val banks = io.physical.toSeq
  val accPipes: Seq[Instance[AccPipe]] = Seq.fill(requestCount)(Instantiate(new AccPipe(frontendParam, memParam)))

  // Per-channel memory trace DPI-C modules to avoid losing simultaneous events
  val mtraces = io.traceMemory.take(requestCount)
  for (mt <- mtraces) {
    mt.is_write   := 0.U
    mt.is_shared  := 0.U
    mt.channel    := 0.U
    mt.thread_id  := 0.U
    mt.rob_id     := 0.U
    mt.inst_id    := 0.U
    mt.vbank_id   := 0.U
    mt.pbank_id   := 0.U
    mt.group_id   := 0.U
    mt.addr       := 0.U
    mt.write_mask := 0.U
    mt.data_lo    := 0.U
    mt.data_hi    := 0.U
    mt.enable     := false.B
  }

  // -----------------------------------------------------------------------------
  // Mapping table
  // -----------------------------------------------------------------------------
  class MappingTableEntry extends Bundle {
    val valid            = Bool()
    val thread_id        = UInt(64.W)
    val vbank_id         = UInt(frontendParam.bank_id_len.W)
    val architectural_id = UInt(frontendParam.bank_id_len.W)
    val is_multi         = Bool()
    val group_id         = UInt(memParam.groupIdWidth.W)
  }

  val mappingTable = RegInit(
    VecInit(Seq.fill(memParam.bankNum)(0.U.asTypeOf(new MappingTableEntry)))
  )

  // The frontend contract reserves [0, max_private_vbank] for private
  // banks. Keep the direct route table at that architectural size.
  val privateVbankCount =
    2 * (memParam.virtualBankCount + (if (gpParam.enable) 2
                                      else
                                        0)) + frontendParam.rob_entries

  require(
    privateVbankCount > 0 && privateVbankCount <= (1 << frontendParam.bank_id_len),
    "Private alias IDs exceed bank_id_len"
  )
  val groupIndexWidth     = log2Up(memParam.bankNum)
  val pbankIndexWidth     = log2Up(memParam.bankNum)
  val privateVbankIdWidth = log2Up(privateVbankCount)

  // A non-multi vbank has exactly one route, so keep that common case as a
  // small indexed table.  Multi-bank mappings retain their per-group entries
  // in mappingTable and use the fallback scan below.  This avoids the much
  // larger 32x24 valid+pbank register matrix while still removing the full
  // mapping scan from the overwhelmingly common single-bank path.
  val singleRouteValid = RegInit(VecInit(Seq.fill(privateVbankCount)(false.B)))

  val singleRoutePbank = RegInit(
    VecInit(Seq.fill(privateVbankCount)(0.U(pbankIndexWidth.W)))
  )

  // The private namespace is bounded by the frontend contract, while
  // multi-bank groups remain represented individually in mappingTable.
  val groupCountByVbank = RegInit(
    VecInit(Seq.fill(privateVbankCount)(0.U(memParam.groupCountWidth.W)))
  )

  val owners = RegInit(VecInit((0 until privateVbankCount).map(i =>
    (i % (memParam.virtualBankCount + (if (gpParam.enable) 2
                                       else 0))).U(
      frontendParam.bank_id_len.W
    )
  )))

  val committed = RegInit(VecInit((0 until memParam.virtualBankCount).map(_.U(
    frontendParam.bank_id_len.W
  ))))

  // Record binding and lifetime changes, including aliases with no active accesses.
  for ((entry, physical) <- mappingTable.zipWithIndex) {
    val live            = entry.valid && true.B
    val current         = entry.valid && committed(entry.architectural_id) === entry.vbank_id
    val previous        = RegNext(entry, 0.U.asTypeOf(entry))
    val previousLive    = RegNext(live, false.B)
    val previousCurrent = RegNext(current, false.B)
    val trace           = io.traceMemory(requestCount + physical)
    trace.is_write   := Mux(entry.valid, 2.U, 3.U)
    trace.is_shared  := 0.U
    trace.channel    := 0.U
    trace.thread_id  := entry.thread_id
    trace.rob_id     := 0.U
    trace.inst_id    := 0.U
    trace.vbank_id   := entry.vbank_id
    trace.pbank_id   := physical.U
    trace.group_id   := entry.group_id
    trace.addr       := entry.architectural_id
    trace.write_mask := 0.U
    trace.data_lo    := live.asUInt
    trace.data_hi    := current.asUInt
    trace.enable     := (entry.valid || previous.valid) &&
      (entry.asUInt =/= previous.asUInt || live =/= previousLive || current =/= previousCurrent)
  }

  val architecturalColumns = RegInit(VecInit(Seq.fill(
    memParam.virtualBankCount
  )(0.U(memParam.groupCountWidth.W))))

  def addEntry(
    thread_id: UInt,
    vbank_id:  UInt,
    pbank_id:  UInt,
    is_multi:  Bool,
    group_id:  UInt
  ): Unit = {
    val entry = mappingTable(pbank_id)
    entry.valid            := true.B
    entry.thread_id        := thread_id
    entry.vbank_id         := vbank_id
    entry.architectural_id := vbank_id
    entry.is_multi         := is_multi
    entry.group_id         := group_id
  }

  def deleteEntry(vbank_id: UInt): Unit = {
    for (i <- 0 until memParam.bankNum) {
      when(mappingTable(i).valid && mappingTable(i).architectural_id === vbank_id) {
        mappingTable(i).valid := false.B
      }
    }
  }

  // -----------------------------------------------------------------------------
  // Default Value
  // -----------------------------------------------------------------------------

  for (i <- 0 until requestCount) {
    accPipes(i).io.mem_req.write <> requests(i).write
    accPipes(i).io.mem_req.read <> requests(i).read
    accPipes(i).io.mem_req.bank_id   := requests(i).bank_id
    accPipes(i).io.mem_req.group_id  := requests(i).group_id
    accPipes(i).io.mem_req.thread_id := requests(i).thread_id

    // Bank-side defaults (only driven when a bank is actually connected)
    accPipes(i).io.sramRead.req.ready  := false.B
    accPipes(i).io.sramRead.resp.valid := false.B
    accPipes(i).io.sramRead.resp.bits  := DontCare

    accPipes(i).io.sramWrite.req.ready  := false.B
    accPipes(i).io.sramWrite.resp.valid := false.B
    accPipes(i).io.sramWrite.resp.bits  := DontCare
  }

  banks.zipWithIndex.foreach {
    case (bank, _) =>
      bank.sramRead.req.valid  := false.B
      bank.sramRead.req.bits   := DontCare
      bank.sramRead.resp.ready := true.B

      bank.sramWrite.req.valid  := false.B
      bank.sramWrite.req.bits   := DontCare
      bank.sramWrite.resp.ready := true.B
      bank.clear                := false.B
      bank.bind                 := false.B
  }
  io.clearBusy := VecInit(banks.map(_.clearing)).asUInt.orR

  // High while a mapping is being installed (see Bank Alloc/Release).
  val mappingBusy = RegInit(false.B)

  val realloc = io.config.bits.group_id === 0.U

  val freePbankMask =
    VecInit(mappingTable.map(entry =>
      !entry.valid ||
        (realloc && entry.architectural_id === io.config.bits.vbank_id)
    ))

  val hasFreePbank = freePbankMask.asUInt.orR

  val configPbank =
    if (memParam.bankLayoutEnable) PriorityEncoder(freePbankMask)
    else (io.config.bits.vbank_id +& io.config.bits.group_id)(pbankIndexWidth - 1, 0)

  val configBankAvailable =
    if (memParam.bankLayoutEnable) hasFreePbank
    else freePbankMask(configPbank)

  if (!memParam.bankLayoutEnable) {
    when(io.config.valid) {
      assert(!io.config.bits.transfer, "Fixed bank layout requires compiler-lowered transfers")
      when(io.config.bits.alloc) {
        assert(
          io.config.bits.vbank_id +& io.config.bits.group_id < memParam.bankNum.U,
          "Fixed bank layout allocation exceeds physical bank count"
        )
      }
    }
    when(io.mapping.valid && io.mapping.bits.allocate) {
      assert(
        io.mapping.bits.bank +& architecturalColumns(
          io.mapping.bits.bank
        ) <= memParam.bankNum.U,
        "Fixed bank alias exceeds physical bank count"
      )
    }
  }
  io.config.ready := !mappingBusy && (!io.config.bits.alloc || configBankAvailable)
  when(io.config.valid && !mappingBusy && io.config.bits.alloc && !configBankAvailable) {
    assert(false.B, "BLT allocation failed: no free physical bank\n")
  }

  // -----------------------------------------------------------------------------
  // Bank Alloc/Release
  // -----------------------------------------------------------------------------

  // A mapping is accepted into registers and installed over the following
  // cycles, one group per cycle. busy stays high until it is installed; the
  // ROB holds the issuing instruction until then, so no consumer ever sees a
  // partially installed mapping.
  val change    = Reg(new BankMapping(frontendParam))
  val nextGroup = RegInit(0.U(log2Up(memParam.bankNum + 1).W))
  io.mapping.busy := mappingBusy

  // An allocation is accepted only when enough banks exist: the unbound ones
  // plus, for an in-place rename, the groups of the alias it replaces.
  val incoming = io.mapping.bits
  val reusable = Mux(incoming.reuse, groupCountByVbank(incoming.previous(privateVbankIdWidth - 1, 0)), 0.U)
  val enough   = PopCount(mappingTable.map(!_.valid)) +& reusable >= architecturalColumns(incoming.bank)
  io.mapping.ready := !mappingBusy && !io.config.valid && (!incoming.allocate ||
    (if (memParam.bankLayoutEnable) enough else true.B))
  when(io.mapping.fire) {
    change      := incoming
    nextGroup   := 0.U
    mappingBusy := true.B
  }

  val columns   = architecturalColumns(change.bank)
  val freeBanks =
    VecInit(mappingTable.map(entry => !entry.valid || (change.reuse && entry.vbank_id === change.previous))).asUInt
  val first     = mappingBusy && nextGroup === 0.U
  when(mappingBusy) {
    nextGroup                                                        := nextGroup + 1.U
    when(!change.allocate || nextGroup + 1.U >= columns)(mappingBusy := false.B)
  }
  when(first && change.allocate) {
    assert(
      columns =/= 0.U && columns <= memParam.bankNum.U,
      "Renamed bank requires an allocated layout"
    )
    owners(change.vbank) := change.bank
  }
  when(first && change.retire) {
    committed(change.bank)             := change.vbank
    singleRouteValid(change.previous)  := false.B
    groupCountByVbank(change.previous) := 0.U
    for (entry <- mappingTable) {
      when(entry.valid && entry.vbank_id === change.previous)(entry.valid := false.B)
    }
  }
  for (bank <- 0 until memParam.bankNum) {
    when(first && change.write && !change.allocate &&
      mappingTable(bank).valid && mappingTable(bank).vbank_id === change.vbank) {
      io.physical(bank).bind := true.B
    }
  }

  when(mappingBusy && change.allocate) {
    val physical =
      if (memParam.bankLayoutEnable) PriorityEncoder(freeBanks)
      else (change.bank + nextGroup)(pbankIndexWidth - 1, 0)
    if (memParam.bankLayoutEnable) {
      assert(freeBanks.orR, "BLT mapping found no free physical bank")
      addEntry(change.threadId, change.vbank, physical, columns > 1.U, nextGroup)
      mappingTable(physical).architectural_id := change.bank
    } else {
      assert(
        mappingTable(physical).valid && mappingTable(physical).architectural_id === change.bank,
        "Fixed bank alias requires its allocated physical slot"
      )
      mappingTable(physical).vbank_id := change.vbank
    }
    for (bank <- 0 until memParam.bankNum) {
      when(physical === bank.U)(io.physical(bank).bind := true.B)
    }
    when(nextGroup === 0.U) {
      singleRoutePbank(change.vbank)  := physical
      singleRouteValid(change.vbank)  := columns === 1.U
      groupCountByVbank(change.vbank) := columns
    }
  }

  when(io.config.fire) {
    val vbank    = io.config.bits.vbank_id
    val vbankIdx = vbank(privateVbankIdWidth - 1, 0)
    when(io.config.bits.transfer) {
      val source       = io.config.bits.source_bank_id
      val sourceIdx    = source(privateVbankIdWidth - 1, 0)
      val sourceCount  = architecturalColumns(source)
      val targetCount  = architecturalColumns(vbank)
      val sourceRoutes = VecInit(mappingTable.map(entry => entry.valid && entry.architectural_id === source))
      val totalCount   = targetCount +& sourceCount
      assert(source =/= vbank && sourceCount =/= 0.U, "Private bank transfer requires a distinct allocated source")
      assert(
        totalCount <= memParam.bankNum.U,
        "Private bank transfer exceeds physical bank count"
      )
      for (entry <- mappingTable) {
        when(entry.valid && entry.architectural_id === source) {
          entry.vbank_id         := vbank
          entry.architectural_id := vbank
          entry.group_id         := targetCount + entry.group_id
          entry.is_multi         := totalCount > 1.U
        }
        when(entry.valid && entry.architectural_id === vbank) {
          entry.vbank_id := vbank
          entry.is_multi := totalCount > 1.U
        }
      }
      singleRouteValid(sourceIdx) := false.B
      singleRouteValid(vbankIdx)                          := totalCount === 1.U
      when(totalCount === 1.U)(singleRoutePbank(vbankIdx) := PriorityEncoder(sourceRoutes.asUInt))
      groupCountByVbank(sourceIdx)                        := 0.U
      groupCountByVbank(vbankIdx)                         := totalCount
      architecturalColumns(source)                        := 0.U
      architecturalColumns(vbank)                         := totalCount
    }.elsewhen(io.config.bits.alloc) {
      val freePbank = configPbank

      for (i <- 0 until memParam.bankNum) {
        io.physical(i).bind := freePbank === i.U
      }

      // The bank hash already treats a newly bound bank as zero, so the clear writes need no hash update.
      for (i <- 0 until memParam.bankNum) {
        when(io.config.bits.clear && freePbank === i.U)(banks(i).clear := true.B)
      }
      // Match bemu mset: realloc of the same vbank frees prior physical banks first.
      // Configer emits one fire per group; only group 0 drops the old mapping.
      when(io.config.bits.group_id === 0.U) {
        deleteEntry(io.config.bits.vbank_id)
        singleRouteValid(vbankIdx) := false.B
      }
      addEntry(
        io.config.bits.thread_id,
        io.config.bits.vbank_id,
        freePbank,
        io.config.bits.is_multi,
        io.config.bits.group_id
      )
      when(!io.config.bits.is_multi) {
        singleRoutePbank(vbankIdx) := freePbank
        singleRouteValid(vbankIdx) := true.B
      }
      groupCountByVbank(vbankIdx) := Mux(io.config.bits.is_multi, io.config.bits.group_id +& 1.U, 1.U)
      architecturalColumns(vbank) := Mux(io.config.bits.is_multi, io.config.bits.group_id +& 1.U, 1.U)
    }.otherwise {
      deleteEntry(io.config.bits.vbank_id)
      singleRouteValid(vbankIdx)  := false.B
      groupCountByVbank(vbankIdx) := 0.U
      architecturalColumns(vbank) := 0.U
    }
  }

  // -----------------------------------------------------------------------------
  // Query interface: return group count for a given vbank_id
  // -----------------------------------------------------------------------------
  val queryVbankId = RegNext(io.query_vbank_id, 0.U)
  val queryVbank   = queryVbankId(privateVbankIdWidth - 1, 0)
  io.query_group_count := RegNext(groupCountByVbank(queryVbank), 0.U)
  for (port <- io.layout) {
    val bank = RegNext(port.bank(privateVbankIdWidth - 1, 0), 0.U)
    port.groups := RegNext(groupCountByVbank(bank), 0.U)
  }

  // -----------------------------------------------------------------------------
  // Connect AccPipe and Banks
  // -----------------------------------------------------------------------------
  def emitTrace(
    ch:        Int,
    isWrite:   UInt,
    pbankId:   UInt,
    addr:      UInt,
    writeMask: UInt,
    dataLo:    UInt,
    dataHi:    UInt,
    en:        Bool
  ): Unit = {
    mtraces(ch).is_write   := isWrite
    mtraces(ch).is_shared  := requests(ch).is_shared.asUInt
    mtraces(ch).channel    := ch.U
    mtraces(ch).thread_id  := requests(ch).thread_id
    mtraces(ch).rob_id     := requests(ch).rob_id
    mtraces(ch).inst_id    := requests(ch).inst_id
    mtraces(ch).vbank_id   := requests(ch).bank_id
    mtraces(ch).pbank_id   := pbankId
    mtraces(ch).group_id   := requests(ch).group_id
    mtraces(ch).addr       := addr
    mtraces(ch).write_mask := writeMask
    mtraces(ch).data_lo    := dataLo
    mtraces(ch).data_hi    := dataHi
    mtraces(ch).enable     := en
  }

  // The route is held for the complete AccPipe transaction. Keep the route
  // vectors in one place so the bank-side command mux and response demux share
  // exactly the same ownership predicate.
  val activeRouteOHs = Seq.fill(requestCount)(Wire(UInt(memParam.bankNum.W)))

  for (i <- 0 until requestCount) {
    val routePbankReg     = RegInit(0.U(pbankIndexWidth.W))
    val routeValidReg     = RegInit(false.B)
    val routePending      = RegInit(false.B)
    val requestVbank      = requests(i).bank_id
    val requestVbankIdx   = requestVbank(privateVbankIdWidth - 1, 0)
    val requestGroup      = requests(i).group_id(groupIndexWidth - 1, 0)
    val singleRoute       = singleRoutePbank(requestVbankIdx)
    val singleValid       = singleRouteValid(requestVbankIdx)
    val multiRouteMatch   = VecInit(mappingTable.map(entry =>
      entry.valid && entry.is_multi &&
        entry.vbank_id === requestVbank && entry.group_id === requestGroup
    ))
    val multiRoute        = PriorityEncoder(multiRouteMatch)
    val multiValid        = multiRouteMatch.asUInt.orR
    val requestRoute      =
      if (memParam.bankLayoutEnable) Mux(singleValid, singleRoute, multiRoute)
      else (owners(requestVbankIdx) + requestGroup)(pbankIndexWidth - 1, 0)
    val requestRouteValid =
      if (memParam.bankLayoutEnable) singleValid || multiValid
      else mappingTable(requestRoute).valid &&
      mappingTable(requestRoute).architectural_id === owners(requestVbankIdx) &&
      mappingTable(requestRoute).group_id === requestGroup
    // Route lookup is deliberately one cycle ahead of the SRAM request.  The
    // route is captured while the AccPipe is idle, then held through the
    // entire request/response lifetime. This removes the live mapping-table
    // lookup from every wide bank-side mux select while preserving one request
    // per channel per cycle once the pipe is running.
    val requestSeen       = requests(i).read.req.valid || requests(i).write.req.valid
    val activeRoute       = routePbankReg
    val requestActive     = routePending || accPipes(i).io.busy
    val activeRouteValid  = routeValidReg && requestActive
    val activeRouteOH     = UIntToOH(activeRoute, memParam.bankNum) &
      Fill(memParam.bankNum, activeRouteValid)

    activeRouteOHs(i) := activeRouteOH

    when(!routePending && !accPipes(i).io.busy && requestSeen && requestRouteValid) {
      routePbankReg := requestRoute
      routeValidReg := true.B
      routePending  := true.B
    }
    when(requests(i).read.req.fire || requests(i).write.req.fire) {
      routePending := false.B
    }

    // Requests are attributed once the selected SRAM accepts them.
    when(accPipes(i).io.sramRead.req.fire) {
      emitTrace(i, 0.U, activeRoute, accPipes(i).io.sramRead.req.bits.addr, 0.U, 0.U, 0.U, true.B)
    }

    // Arrival trace: the write has crossed arbitration and is accepted by the
    // selected SPM SRAM port. Keep attribution metadata from the same AccPipe
    // transaction and data from the bank-side request.
    when(accPipes(i).io.sramWrite.req.fire) {
      emitTrace(
        i,
        1.U,
        activeRoute,
        accPipes(i).io.sramWrite.req.bits.addr,
        accPipes(i).io.sramWrite.req.bits.mask.asUInt,
        accPipes(i).io.sramWrite.req.bits.data(63, 0),
        accPipes(i).io.sramWrite.req.bits.data(127, 64),
        true.B
      )
    }

  }

  // Fixed GP request slots share the physical-bank arbitration with ordinary
  // channels. Capture the accepted request's owner for the one-cycle SRAM
  // response; a stalled requester must never receive another owner's response.
  val responseRoutes    =
    Seq.fill(requestCount)(WireDefault(0.U(memParam.bankNum.W)))
  val responseRouteBits =
    Seq.fill(requestCount)(Wire(Vec(memParam.bankNum, Bool())))

  for (j <- 0 until memParam.bankNum) {
    val arb = Instantiate(new RRArbiter(
      new BankRequest(memParam),
      requestCount
    ))

    for (i <- 0 until requestCount) {
      val pipe = accPipes(i)
      arb.io.in(i).valid      := activeRouteOHs(i)(j) &&
        (pipe.io.sramRead.req.valid || pipe.io.sramWrite.req.valid)
      arb.io.in(i).bits.write := pipe.io.sramWrite.req.valid
      arb.io.in(i).bits.addr  := Mux(
        pipe.io.sramWrite.req.valid,
        pipe.io.sramWrite.req.bits.addr,
        pipe.io.sramRead.req.bits.addr
      )
      arb.io.in(i).bits.data  := pipe.io.sramWrite.req.bits.data
      arb.io.in(i).bits.mask  := pipe.io.sramWrite.req.bits.mask
      when(activeRouteOHs(i)(j)) {
        pipe.io.sramRead.req.ready  := arb.io.in(i).ready && !arb.io.in(i).bits.write
        pipe.io.sramWrite.req.ready := arb.io.in(i).ready && arb.io.in(i).bits.write
      }
      responseRouteBits(i)(j) := RegNext(arb.io.in(i).fire, false.B)
    }

    val bank = banks(j)
    bank.sramRead.req.valid      := arb.io.out.valid && !arb.io.out.bits.write
    bank.sramRead.req.bits.addr  := arb.io.out.bits.addr
    bank.sramWrite.req.valid     := arb.io.out.valid && arb.io.out.bits.write
    bank.sramWrite.req.bits.addr := arb.io.out.bits.addr
    bank.sramWrite.req.bits.data := arb.io.out.bits.data
    bank.sramWrite.req.bits.mask := arb.io.out.bits.mask
    arb.io.out.ready             := Mux(arb.io.out.bits.write, bank.sramWrite.req.ready, bank.sramRead.req.ready)

    bank.sramRead.resp.ready  := VecInit((0 until requestCount).map { i =>
      responseRouteBits(i)(j) && accPipes(i).io.sramRead.resp.ready
    }).asUInt.orR
    bank.sramWrite.resp.ready := VecInit((0 until requestCount).map { i =>
      responseRouteBits(i)(j) && accPipes(i).io.sramWrite.resp.ready
    }).asUInt.orR
    when(bank.sramRead.resp.valid)(assert(bank.sramRead.resp.ready))
    when(bank.sramWrite.resp.valid)(assert(bank.sramWrite.resp.ready))
  }

  for (i <- 0 until requestCount) {
    responseRoutes(i) := responseRouteBits(i).asUInt
    val readRespSel  = VecInit((0 until memParam.bankNum).map { j =>
      responseRoutes(i)(j) && banks(j).sramRead.resp.valid
    }).asUInt
    val writeRespSel = VecInit((0 until memParam.bankNum).map { j =>
      responseRoutes(i)(j) && banks(j).sramWrite.resp.valid
    }).asUInt
    accPipes(i).io.sramRead.resp.valid     := readRespSel.orR
    accPipes(i).io.sramRead.resp.bits.data := (0 until memParam.bankNum).map {
      j =>
        Fill(memParam.bankWidth, responseRoutes(i)(j)) & banks(
          j
        ).sramRead.resp.bits.data
    }.reduce(_ | _)
    accPipes(i).io.sramWrite.resp.valid    := writeRespSel.orR
    accPipes(i).io.sramWrite.resp.bits.ok  := writeRespSel.orR
  }

  io.bank_hashes.foreach { states =>
    for (i <- 0 until memParam.bankNum) {
      states(i).valid      := mappingTable(i).valid
      states(i).threadId   := mappingTable(i).thread_id
      states(i).vbankId    := (if (memParam.bankLayoutEnable)
                              mappingTable(i).vbank_id
                            else mappingTable(i).architectural_id)
      states(i).pbankId    := i.U
      states(i).groupId    := mappingTable(i).group_id
      states(i).statusHash := io.hashes.get(i)
    }
  }
}
