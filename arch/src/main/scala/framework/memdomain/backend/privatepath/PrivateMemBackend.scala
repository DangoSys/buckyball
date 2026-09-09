package framework.memdomain.backend.privatepath

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.memdomain.frontend.mem.MemConfigerIO
import framework.memdomain.backend.{MTraceDPI, MemRequestIO}
import framework.memdomain.backend.accpipe.AccPipe
import framework.memdomain.backend.banks.SramBank
import framework.top.GlobalConfig

@instantiable
class PrivateMemBackend(val b: GlobalConfig) extends Module {

  @public
  val io = IO(new Bundle {
    val mem_req = Vec(b.memDomain.bankChannel, Flipped(new MemRequestIO(b)))
    val config  = Flipped(Decoupled(new MemConfigerIO(b)))

    // Query interface for frontend to get group count
    val query_vbank_id    = Input(UInt(8.W))
    val query_group_count = Output(UInt(log2Up(b.memDomain.bankNum + 1).W))
  })

  val banks:    Seq[Instance[SramBank]] = Seq.fill(b.memDomain.bankNum)(Instantiate(new SramBank(b)))
  val accPipes: Seq[Instance[AccPipe]]  = Seq.fill(b.memDomain.bankChannel)(Instantiate(new AccPipe(b)))

  // Per-channel memory trace DPI-C modules to avoid losing simultaneous events
  val mtraces = Seq.fill(b.memDomain.bankChannel)(Module(new MTraceDPI))
  for (mt <- mtraces) {
    mt.io.clock      := clock
    mt.io.reset      := reset.asBool
    mt.io.is_write   := 0.U
    mt.io.is_shared  := 0.U
    mt.io.channel    := 0.U
    mt.io.hart_id    := 0.U
    mt.io.rob_id     := 0.U
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
    val vbank_id = UInt(5.W)
    val is_multi = Bool()
    val group_id = UInt(log2Up(b.memDomain.bankNum).W)
  }

  val mappingTable              = RegInit(VecInit(Seq.fill(b.memDomain.bankNum)(0.U.asTypeOf(new MappingTableEntry))))
  // The frontend contract reserves [0, vbank_id_upper_bound] for private
  // banks.  Keep the direct route table at that architectural size instead
  // of materializing all 32 encodings of the five-bit wire field.
  private val privateVbankCount = b.frontend.vbank_id_upper_bound + 1
  require(
    privateVbankCount > 0 && privateVbankCount <= (1 << 5),
    s"private vbank table size ($privateVbankCount) must fit the five-bit vbank namespace"
  )
  private val groupIndexWidth   = log2Up(b.memDomain.bankNum)
  private val pbankIndexWidth   = log2Up(b.memDomain.bankNum)

  // A non-multi vbank has exactly one route, so keep that common case as a
  // small indexed table.  Multi-bank mappings retain their per-group entries
  // in mappingTable and use the fallback scan below.  This avoids the much
  // larger 32x24 valid+pbank register matrix while still removing the full
  // mapping scan from the overwhelmingly common single-bank path.
  val singleRouteValid = RegInit(VecInit(Seq.fill(privateVbankCount)(false.B)))

  val singleRoutePbank = RegInit(
    VecInit(Seq.fill(privateVbankCount)(0.U(pbankIndexWidth.W)))
  )

  // Virtual bank ids are encoded in the five-bit mapping-table field. The
  // private namespace is bounded by the frontend contract, while multi-bank
  // groups remain represented individually in mappingTable.
  val groupCountByVbank = RegInit(
    VecInit(Seq.fill(privateVbankCount)(0.U(log2Up(b.memDomain.bankNum + 1).W)))
  )

  def addEntry(
    vbank_id: UInt,
    pbank_id: UInt,
    is_multi: Bool,
    group_id: UInt
  ): Unit = {
    val entry = mappingTable(pbank_id)
    entry.valid    := true.B
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

  def getFreePbankId(): UInt = {
    val hasFree     = mappingTable.map(_.valid === false.B).reduce(_ || _)
    when(!hasFree) {
      assert(false.B, "PrivateMemBackend allocation failed: no free physical bank\n")
    }
    val freePbankId = mappingTable.indexWhere(_.valid === false.B)
    freePbankId
  }

  // -----------------------------------------------------------------------------
  // Default Value
  // -----------------------------------------------------------------------------

  for (i <- 0 until b.memDomain.bankChannel) {
    accPipes(i).io.mem_req.write <> io.mem_req(i).write
    accPipes(i).io.mem_req.read <> io.mem_req(i).read
    accPipes(i).io.mem_req.bank_id   := io.mem_req(i).bank_id
    accPipes(i).io.mem_req.group_id  := io.mem_req(i).group_id
    accPipes(i).io.mem_req.is_shared := io.mem_req(i).is_shared
    accPipes(i).io.mem_req.hart_id   := io.mem_req(i).hart_id
    accPipes(i).io.mem_req.rob_id    := io.mem_req(i).rob_id

    // Bank-side defaults (only driven when a bank is actually connected)
    accPipes(i).io.sramRead.req.ready  := false.B
    accPipes(i).io.sramRead.resp.valid := false.B
    accPipes(i).io.sramRead.resp.bits  := DontCare

    accPipes(i).io.sramWrite.req.ready  := false.B
    accPipes(i).io.sramWrite.resp.valid := false.B
    accPipes(i).io.sramWrite.resp.bits  := DontCare

    accPipes(i).io.is_multi := false.B
  }

  banks.zipWithIndex.foreach {
    case (bank, _) =>
      bank.io.sramRead.req.valid  := false.B
      bank.io.sramRead.req.bits   := DontCare
      bank.io.sramRead.resp.ready := true.B

      bank.io.sramWrite.req.valid  := false.B
      bank.io.sramWrite.req.bits   := DontCare
      bank.io.sramWrite.resp.ready := true.B
  }

  io.config.ready := true.B

  // -----------------------------------------------------------------------------
  // Bank Alloc/Release
  // -----------------------------------------------------------------------------

  when(io.config.fire) {
    val vbank = io.config.bits.vbank_id(4, 0)
    when(io.config.bits.alloc) {
      val freePbank = getFreePbankId()
      // Match bemu mset: realloc of the same vbank frees prior physical banks first.
      // MemConfiger emits one fire per group; only group 0 drops the old mapping.
      when(io.config.bits.group_id === 0.U) {
        deleteEntry(io.config.bits.vbank_id)
        singleRouteValid(vbank) := false.B
      }
      addEntry(
        io.config.bits.vbank_id,
        freePbank,
        io.config.bits.is_multi,
        io.config.bits.group_id
      )
      when(!io.config.bits.is_multi) {
        singleRoutePbank(vbank) := freePbank
        singleRouteValid(vbank) := true.B
      }
      groupCountByVbank(vbank) := Mux(io.config.bits.is_multi, io.config.bits.group_id +& 1.U, 1.U)
    }.otherwise {
      deleteEntry(io.config.bits.vbank_id)
      singleRouteValid(vbank)  := false.B
      groupCountByVbank(vbank) := 0.U
    }
  }

  // -----------------------------------------------------------------------------
  // Query interface: return group count for a given vbank_id
  // -----------------------------------------------------------------------------
  val queryVbankId = RegNext(io.query_vbank_id, 0.U)
  val queryVbank   = queryVbankId(4, 0)
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
    mtraces(ch).io.is_shared  := io.mem_req(ch).is_shared.asUInt
    mtraces(ch).io.channel    := ch.U
    mtraces(ch).io.hart_id    := io.mem_req(ch).hart_id
    mtraces(ch).io.rob_id     := io.mem_req(ch).rob_id
    mtraces(ch).io.vbank_id   := io.mem_req(ch).bank_id
    mtraces(ch).io.pbank_id   := pbankId
    mtraces(ch).io.group_id   := io.mem_req(ch).group_id
    mtraces(ch).io.addr       := addr
    mtraces(ch).io.write_mask := writeMask
    mtraces(ch).io.data_lo    := dataLo
    mtraces(ch).io.data_hi    := dataHi
    mtraces(ch).io.enable     := en
  }

  // The route is held for the complete AccPipe transaction. Keep the route
  // vectors in one place so the bank-side command mux and response demux share
  // exactly the same ownership predicate.
  val activeRouteOHs    = Seq.fill(b.memDomain.bankChannel)(Wire(UInt(b.memDomain.bankNum.W)))
  val channelReqValid   = Seq.fill(b.memDomain.bankChannel)(Wire(Bool()))
  val channelReqIsWrite = Seq.fill(b.memDomain.bankChannel)(Wire(Bool()))
  val channelReqAddr    = Seq.fill(b.memDomain.bankChannel)(Wire(UInt(log2Ceil(b.memDomain.bankEntries).W)))
  val channelReqData    = Seq.fill(b.memDomain.bankChannel)(Wire(UInt(b.memDomain.bankWidth.W)))
  val channelReqMask    = Seq.fill(b.memDomain.bankChannel)(Wire(Vec(b.memDomain.bankMaskLen, Bool())))

  for (i <- 0 until b.memDomain.bankChannel) {
    val routePbankReg     = RegInit(0.U(pbankIndexWidth.W))
    val routeValidReg     = RegInit(false.B)
    val routePending      = RegInit(false.B)
    // The private vbank namespace is five bits wide even when a chip has
    // fewer than 32 physical banks (e.g. Toy/Goban use a four-bit bank ID).
    // Zero-extend the request ID instead of slicing a potentially narrow bus.
    val requestVbank      = io.mem_req(i).bank_id.pad(5)
    val requestGroup      = io.mem_req(i).group_id(groupIndexWidth - 1, 0)
    val singleRoute       = singleRoutePbank(requestVbank)
    val singleValid       = singleRouteValid(requestVbank)
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
    val requestSeen       = io.mem_req(i).read.req.valid || io.mem_req(i).write.req.valid
    val activeRoute       = routePbankReg
    val requestActive     = routePending || accPipes(i).io.busy
    val activeRouteValid  = routeValidReg && requestActive
    val activeRouteOH     = UIntToOH(activeRoute, b.memDomain.bankNum) &
      Fill(b.memDomain.bankNum, activeRouteValid)

    activeRouteOHs(i)    := activeRouteOH
    channelReqValid(i)   := accPipes(i).io.sramRead.req.valid ||
      accPipes(i).io.sramWrite.req.valid
    channelReqIsWrite(i) := accPipes(i).io.sramWrite.req.valid
    channelReqAddr(i)    := Mux(
      channelReqIsWrite(i),
      accPipes(i).io.sramWrite.req.bits.addr,
      accPipes(i).io.sramRead.req.bits.addr
    )
    channelReqData(i)    := accPipes(i).io.sramWrite.req.bits.data
    channelReqMask(i)    := accPipes(i).io.sramWrite.req.bits.mask

    // SramBank is a pure single-port SRAM: the selected operation is always
    // accepted when its route is active. Using the held route directly avoids
    // feeding 24 bank-ready signals back through every channel.
    accPipes(i).io.sramRead.req.ready  := activeRouteValid
    accPipes(i).io.sramWrite.req.ready := activeRouteValid

    when(!routePending && !accPipes(i).io.busy && requestSeen && requestRouteValid) {
      routePbankReg := requestRoute
      routeValidReg := true.B
      routePending  := true.B
    }
    when(io.mem_req(i).read.req.fire || io.mem_req(i).write.req.fire) {
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

  // Bank-side command crossbar. A physical bank is single-port, so both read
  // and write requests share one selected channel/address path. The operation
  // type then gates the corresponding SRAM request valid.
  for (j <- 0 until b.memDomain.bankNum) {
    val selectedReq      = VecInit((0 until b.memDomain.bankChannel).map { i =>
      activeRouteOHs(i)(j) && channelReqValid(i)
    }).asUInt
    val selectedReqValid = selectedReq.orR
    // The scheduler guarantees one live owner per physical bank. Express the
    // one-hot selection as AND/OR wiring instead of a priority mux tree; this
    // keeps the wide address/data path shallow and lets synthesis share the
    // route gates across fields.
    def oneHotOr(width: Int, values: Seq[UInt]): UInt =
      values.zipWithIndex
        .map { case (value, i) => Fill(width, selectedReq(i)) & value }
        .reduce(_ | _)

    val selectedReqIsWrite = oneHotOr(1, channelReqIsWrite).asBool
    val selectedReqAddr    = oneHotOr(log2Ceil(b.memDomain.bankEntries), channelReqAddr)
    val selectedReqData    = oneHotOr(b.memDomain.bankWidth, channelReqData)
    val selectedReqMask    = VecInit((0 until b.memDomain.bankMaskLen).map { k =>
      oneHotOr(1, channelReqMask.map(_(k))).asBool
    })

    banks(j).io.sramRead.req.valid      := selectedReqValid && !selectedReqIsWrite
    banks(j).io.sramRead.req.bits.addr  := selectedReqAddr
    banks(j).io.sramWrite.req.valid     := selectedReqValid && selectedReqIsWrite
    banks(j).io.sramWrite.req.bits.addr := selectedReqAddr
    banks(j).io.sramWrite.req.bits.data := selectedReqData
    banks(j).io.sramWrite.req.bits.mask := selectedReqMask

    // Only the channel holding this bank can consume its one-cycle response.
    banks(j).io.sramRead.resp.ready  := VecInit((0 until b.memDomain.bankChannel).map { i =>
      activeRouteOHs(i)(j) && accPipes(i).io.sramRead.resp.ready
    }).asUInt.orR
    banks(j).io.sramWrite.resp.ready := VecInit((0 until b.memDomain.bankChannel).map { i =>
      activeRouteOHs(i)(j) && accPipes(i).io.sramWrite.resp.ready
    }).asUInt.orR
  }

  // Response demux. Read data remains the only wide bank-to-channel mux; the
  // request side above shares its address/data path between read and write.
  for (i <- 0 until b.memDomain.bankChannel) {
    val readRespSel  = VecInit((0 until b.memDomain.bankNum).map { j =>
      activeRouteOHs(i)(j) && banks(j).io.sramRead.resp.valid
    }).asUInt
    val writeRespSel = VecInit((0 until b.memDomain.bankNum).map { j =>
      activeRouteOHs(i)(j) && banks(j).io.sramWrite.resp.valid
    }).asUInt

    accPipes(i).io.sramRead.resp.valid     := readRespSel.orR
    accPipes(i).io.sramRead.resp.bits.data := (0 until b.memDomain.bankNum).map { j =>
      Fill(b.memDomain.bankWidth, activeRouteOHs(i)(j)) & banks(j).io.sramRead.resp.bits.data
    }.reduce(_ | _)
    accPipes(i).io.sramWrite.resp.valid    := writeRespSel.orR
    accPipes(i).io.sramWrite.resp.bits.ok  := writeRespSel.orR
  }
}
