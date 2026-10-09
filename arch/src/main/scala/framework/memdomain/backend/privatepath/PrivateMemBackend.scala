package framework.memdomain.backend.privatepath

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.memdomain.frontend.mem.MemConfigerIO
import framework.memdomain.backend.{MTraceDPI, MemRequestIO}
import sims.hash.BankHashMonitor
import framework.memdomain.backend.accpipe.AccPipe
import framework.memdomain.backend.banks.SramBank
import framework.memdomain.backend.banks.btrace.PhysicalBankHash
import framework.top.GlobalConfig

@instantiable
class PrivateMemBackend(val b: GlobalConfig) extends Module {
  val kernelRequests = if (b.rvv.enable) 2 * b.rvv.memoryPorts else 0
  val requestCount   = b.memDomain.bankChannel + kernelRequests

  @public
  val io = IO(new Bundle {
    val mem_req    = Vec(b.memDomain.bankChannel, Flipped(new MemRequestIO(b)))
    val kernel_req = Vec(kernelRequests, Flipped(new MemRequestIO(b)))
    val config     = Flipped(Decoupled(new MemConfigerIO(b)))

    // Query interface for frontend to get group count
    val query_vbank_id    = Input(UInt(b.memDomain.vbankIdWidth.W))
    val query_group_count = Output(UInt(b.memDomain.groupCountWidth.W))
    val clearBusy         = Output(Bool())
    val bank_hashes       = if (b.sim.diffTest) Some(Output(Vec(b.memDomain.bankNum, new PhysicalBankHash(b)))) else None
  })

  val requests = io.mem_req.toSeq ++ io.kernel_req.toSeq

  val banks:    Seq[Instance[SramBank]] = Seq.fill(b.memDomain.bankNum)(Instantiate(new SramBank(b)))
  val accPipes: Seq[Instance[AccPipe]]  = Seq.fill(requestCount)(Instantiate(new AccPipe(b)))

  val hashMonitors: Option[Seq[Instance[BankHashMonitor]]] =
    if (b.sim.diffTest) {
      Some(Seq.fill(b.memDomain.bankNum)(Instantiate(new BankHashMonitor(b))))
    } else {
      None
    }

  // Per-channel memory trace DPI-C modules to avoid losing simultaneous events
  val mtraces = Seq.fill(requestCount)(Module(new MTraceDPI))
  for (mt <- mtraces) {
    mt.io.clock      := clock
    mt.io.reset      := reset.asBool
    mt.io.is_write   := 0.U
    mt.io.is_shared  := 0.U
    mt.io.channel    := 0.U
    mt.io.hart_id    := 0.U
    mt.io.rob_id     := 0.U
    mt.io.inst_id    := 0.U
    mt.io.vbank_id   := 0.U
    mt.io.pbank_id   := 0.U
    mt.io.group_id   := 0.U
    mt.io.addr       := 0.U
    mt.io.write_mask := 0.U
    mt.io.data_lo    := 0.U
    mt.io.data_hi    := 0.U
    mt.io.enable     := false.B
  }

  // -----------------------------------------------------------------------------
  // Mapping table
  // -----------------------------------------------------------------------------
  class MappingTableEntry extends Bundle {
    val valid    = Bool()
    val hart_id  = UInt(b.tile.xLen.W)
    val vbank_id = UInt(b.memDomain.vbankIdWidth.W)
    val is_multi = Bool()
    val group_id = UInt(b.memDomain.groupIdWidth.W)
  }

  val mappingTable        = RegInit(VecInit(Seq.fill(b.memDomain.bankNum)(0.U.asTypeOf(new MappingTableEntry))))
  // The frontend contract reserves [0, vbank_id_upper_bound] for private
  // banks. Keep the direct route table at that architectural size.
  val privateVbankCount   = b.frontend.vbank_id_upper_bound + 1
  require(
    privateVbankCount > 0 && privateVbankCount <= b.memDomain.virtualBankCount,
    s"private vbank table size ($privateVbankCount) exceeds virtualBankCount(${b.memDomain.virtualBankCount})"
  )
  val groupIndexWidth     = log2Up(b.memDomain.bankNum)
  val pbankIndexWidth     = log2Up(b.memDomain.bankNum)
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
    VecInit(Seq.fill(privateVbankCount)(0.U(b.memDomain.groupCountWidth.W)))
  )

  def addEntry(
    hart_id:  UInt,
    vbank_id: UInt,
    pbank_id: UInt,
    is_multi: Bool,
    group_id: UInt
  ): Unit = {
    val entry = mappingTable(pbank_id)
    entry.valid    := true.B
    entry.hart_id  := hart_id
    entry.vbank_id := vbank_id
    entry.is_multi := is_multi
    entry.group_id := group_id
  }

  def deleteEntry(vbank_id: UInt): Unit = {
    for (i <- 0 until b.memDomain.bankNum) {
      when(mappingTable(i).valid && mappingTable(i).vbank_id === vbank_id) {
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
    accPipes(i).io.mem_req.is_shared := requests(i).is_shared
    accPipes(i).io.mem_req.hart_id   := requests(i).hart_id
    accPipes(i).io.mem_req.rob_id    := requests(i).rob_id
    accPipes(i).io.mem_req.inst_id   := requests(i).inst_id

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
      bank.io.sramRead.req.valid  := false.B
      bank.io.sramRead.req.bits   := DontCare
      bank.io.sramRead.resp.ready := true.B

      bank.io.sramWrite.req.valid  := false.B
      bank.io.sramWrite.req.bits   := DontCare
      bank.io.sramWrite.resp.ready := true.B
      bank.io.clear                := false.B
  }
  io.clearBusy := VecInit(banks.map(_.io.clearing)).asUInt.orR

  val realloc       = io.config.bits.group_id === 0.U
  val freePbankMask =
    VecInit(mappingTable.map(entry => !entry.valid || (realloc && entry.vbank_id === io.config.bits.vbank_id)))
  val hasFreePbank  = freePbankMask.asUInt.orR
  io.config.ready := !io.config.bits.alloc || hasFreePbank
  when(io.config.valid && io.config.bits.alloc && !hasFreePbank) {
    assert(false.B, "PrivateMemBackend allocation failed: no free physical bank\n")
  }

  // -----------------------------------------------------------------------------
  // Bank Alloc/Release
  // -----------------------------------------------------------------------------

  hashMonitors.foreach(_.foreach(_.io.bind := false.B))

  when(io.config.fire) {
    val vbank    = io.config.bits.vbank_id
    val vbankIdx = vbank(privateVbankIdWidth - 1, 0)
    when(io.config.bits.transfer) {
      val source      = io.config.bits.source_bank_id
      val sourceIdx   = source(privateVbankIdWidth - 1, 0)
      val sourceCount = groupCountByVbank(sourceIdx)
      val targetCount = groupCountByVbank(vbankIdx)
      val totalCount  = targetCount +& sourceCount
      assert(source =/= vbank && sourceCount =/= 0.U, "Private bank transfer requires a distinct allocated source")
      assert(totalCount <= b.memDomain.bankNum.U, "Private bank transfer exceeds physical bank count")
      for (entry <- mappingTable) {
        when(entry.valid && entry.vbank_id === source) {
          entry.vbank_id := vbank
          entry.group_id := targetCount + entry.group_id
          entry.is_multi := totalCount > 1.U
        }
        when(entry.valid && entry.vbank_id === vbank) {
          entry.is_multi := totalCount > 1.U
        }
      }
      singleRouteValid(sourceIdx) := false.B
      singleRouteValid(vbankIdx)                          := totalCount === 1.U
      when(totalCount === 1.U)(singleRoutePbank(vbankIdx) := singleRoutePbank(sourceIdx))
      groupCountByVbank(sourceIdx)                        := 0.U
      groupCountByVbank(vbankIdx)                         := totalCount
    }.elsewhen(io.config.bits.alloc) {
      val freePbank = PriorityEncoder(freePbankMask)
      hashMonitors.foreach { hashes =>
        for (i <- 0 until b.memDomain.bankNum) {
          hashes(i).io.bind := freePbank === i.U
        }
      }
      // The bank hash already treats a newly bound bank as zero, so the clear writes need no hash update.
      for (i <- 0 until b.memDomain.bankNum) {
        when(io.config.bits.clear && freePbank === i.U)(banks(i).io.clear := true.B)
      }
      // Match bemu mset: realloc of the same vbank frees prior physical banks first.
      // MemConfiger emits one fire per group; only group 0 drops the old mapping.
      when(io.config.bits.group_id === 0.U) {
        deleteEntry(io.config.bits.vbank_id)
        singleRouteValid(vbankIdx) := false.B
      }
      addEntry(
        io.config.bits.hart_id,
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
    }.otherwise {
      deleteEntry(io.config.bits.vbank_id)
      singleRouteValid(vbankIdx)  := false.B
      groupCountByVbank(vbankIdx) := 0.U
    }
  }

  // -----------------------------------------------------------------------------
  // Query interface: return group count for a given vbank_id
  // -----------------------------------------------------------------------------
  val queryVbankId = RegNext(io.query_vbank_id, 0.U)
  val queryVbank   = queryVbankId(privateVbankIdWidth - 1, 0)
  io.query_group_count := RegNext(groupCountByVbank(queryVbank), 0.U)

  // -----------------------------------------------------------------------------
  // Connect AccPipe and Banks
  // -----------------------------------------------------------------------------
  private def emitTrace(
    ch:        Int,
    isWrite:   UInt,
    pbankId:   UInt,
    addr:      UInt,
    writeMask: UInt,
    dataLo:    UInt,
    dataHi:    UInt,
    en:        Bool
  ): Unit = {
    mtraces(ch).io.is_write   := isWrite
    mtraces(ch).io.is_shared  := requests(ch).is_shared.asUInt
    mtraces(ch).io.channel    := ch.U
    mtraces(ch).io.hart_id    := requests(ch).hart_id
    mtraces(ch).io.rob_id     := requests(ch).rob_id
    mtraces(ch).io.inst_id    := requests(ch).inst_id
    mtraces(ch).io.vbank_id   := requests(ch).bank_id
    mtraces(ch).io.pbank_id   := pbankId
    mtraces(ch).io.group_id   := requests(ch).group_id
    mtraces(ch).io.addr       := addr
    mtraces(ch).io.write_mask := writeMask
    mtraces(ch).io.data_lo    := dataLo
    mtraces(ch).io.data_hi    := dataHi
    mtraces(ch).io.enable     := en
  }

  // The route is held for the complete AccPipe transaction. Keep the route
  // vectors in one place so the bank-side command mux and response demux share
  // exactly the same ownership predicate.
  val activeRouteOHs = Seq.fill(requestCount)(Wire(UInt(b.memDomain.bankNum.W)))

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
    val requestRoute      = Mux(singleValid, singleRoute, multiRoute)
    val requestRouteValid = singleValid || multiValid
    // Route lookup is deliberately one cycle ahead of the SRAM request.  The
    // route is captured while the AccPipe is idle, then held through the
    // entire request/response lifetime. This removes the live mapping-table
    // lookup from every wide bank-side mux select while preserving one request
    // per channel per cycle once the pipe is running.
    val requestSeen       = requests(i).read.req.valid || requests(i).write.req.valid
    val activeRoute       = routePbankReg
    val requestActive     = routePending || accPipes(i).io.busy
    val activeRouteValid  = routeValidReg && requestActive
    val activeRouteOH     = UIntToOH(activeRoute, b.memDomain.bankNum) &
      Fill(b.memDomain.bankNum, activeRouteValid)

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

  // Fixed RVV request slots share the physical-bank arbitration with ordinary
  // channels. Capture the accepted request's owner for the one-cycle SRAM
  // response; a stalled requester must never receive another owner's response.
  val responseRoutes    = Seq.fill(requestCount)(WireDefault(0.U(b.memDomain.bankNum.W)))
  val responseRouteBits = Seq.fill(requestCount)(Wire(Vec(b.memDomain.bankNum, Bool())))

  for (j <- 0 until b.memDomain.bankNum) {
    val arb = Module(new RRArbiter(
      new Bundle {
        val write = Bool()
        val addr  = UInt(16.W)
        val data  = UInt(b.memDomain.bankWidth.W)
        val mask  = Vec(b.memDomain.bankMaskLen, Bool())
      },
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
    bank.io.sramRead.req.valid      := arb.io.out.valid && !arb.io.out.bits.write
    bank.io.sramRead.req.bits.addr  := arb.io.out.bits.addr
    bank.io.sramWrite.req.valid     := arb.io.out.valid && arb.io.out.bits.write
    bank.io.sramWrite.req.bits.addr := arb.io.out.bits.addr
    bank.io.sramWrite.req.bits.data := arb.io.out.bits.data
    bank.io.sramWrite.req.bits.mask := arb.io.out.bits.mask
    arb.io.out.ready                := Mux(arb.io.out.bits.write, bank.io.sramWrite.req.ready, bank.io.sramRead.req.ready)

    hashMonitors.foreach { hashes =>
      val monitor = hashes(j)
      bank.io.sramWrite.req.valid := arb.io.out.valid && arb.io.out.bits.write && monitor.io.write.ready
      monitor.io.write.valid      := arb.io.out.valid && arb.io.out.bits.write && bank.io.sramWrite.req.ready
      arb.io.out.ready            := Mux(
        arb.io.out.bits.write,
        bank.io.sramWrite.req.ready && monitor.io.write.ready,
        bank.io.sramRead.req.ready
      )
      monitor.io.write.bits.addr  := arb.io.out.bits.addr
      monitor.io.write.bits.mask  := arb.io.out.bits.mask
      monitor.io.write.bits.data  := arb.io.out.bits.data
    }
    bank.io.sramRead.resp.ready  := VecInit((0 until requestCount).map { i =>
      responseRouteBits(i)(j) && accPipes(i).io.sramRead.resp.ready
    }).asUInt.orR
    bank.io.sramWrite.resp.ready := VecInit((0 until requestCount).map { i =>
      responseRouteBits(i)(j) && accPipes(i).io.sramWrite.resp.ready
    }).asUInt.orR
    when(bank.io.sramRead.resp.valid)(assert(bank.io.sramRead.resp.ready))
    when(bank.io.sramWrite.resp.valid)(assert(bank.io.sramWrite.resp.ready))
  }

  for (i <- 0 until requestCount) {
    responseRoutes(i) := responseRouteBits(i).asUInt
    val readRespSel  = VecInit((0 until b.memDomain.bankNum).map { j =>
      responseRoutes(i)(j) && banks(j).io.sramRead.resp.valid
    }).asUInt
    val writeRespSel = VecInit((0 until b.memDomain.bankNum).map { j =>
      responseRoutes(i)(j) && banks(j).io.sramWrite.resp.valid
    }).asUInt
    accPipes(i).io.sramRead.resp.valid     := readRespSel.orR
    accPipes(i).io.sramRead.resp.bits.data := (0 until b.memDomain.bankNum).map { j =>
      Fill(b.memDomain.bankWidth, responseRoutes(i)(j)) & banks(j).io.sramRead.resp.bits.data
    }.reduce(_ | _)
    accPipes(i).io.sramWrite.resp.valid    := writeRespSel.orR
    accPipes(i).io.sramWrite.resp.bits.ok  := writeRespSel.orR
  }

  io.bank_hashes.foreach { states =>
    for (i <- 0 until b.memDomain.bankNum) {
      states(i).valid      := mappingTable(i).valid
      states(i).hartId     := mappingTable(i).hart_id
      states(i).vbankId    := mappingTable(i).vbank_id
      states(i).pbankId    := i.U
      states(i).groupId    := mappingTable(i).group_id
      states(i).statusHash := hashMonitors.get(i).io.statusHash
    }
  }
}
