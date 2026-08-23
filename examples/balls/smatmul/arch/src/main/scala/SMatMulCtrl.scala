package framework.balldomain.prototype.systolicarray

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.balldomain.rs.{BallRsComplete, BallRsIssue}
import framework.top.GlobalConfig

@instantiable
class SystolicArrayCtrl(val b: GlobalConfig) extends Module {

  /**
   * 脉动阵列控制器。
   *
   * 一条命令计算一块 cols<=16 的 C panel。Ctrl 按 16x16 tile 沿 M、K 发射；
   * N 只有一个 tile。K 维在指令内部累加，写回按输出行顺序给出 C 地址。
   *
   * OS: 每个 M tile 走完整条 K 链再切下一 M tile。
   * WS: 一批 M tile 共享驻留 B；K 在 batch 内层复用权重。
   */
  private val tile         = SystolicArrayConst.Tile
  private val abRows       = b.memDomain.bankEntries
  private val groupWidth   = log2Up(b.memDomain.bankNum)
  private val addrWidth    = log2Up(b.memDomain.bankEntries)
  private val iterLen      = b.frontend.iter_len
  private val countWidth   = 32
  private val wsReuseTiles = SystolicArrayConst.wsReuseTiles(b)

  private val osFunct7 = b.ballDomain.ballISA
    .find(_.mnemonic == "SMATMUL_OS")
    .map(_.funct7)
    .getOrElse(throw new IllegalArgumentException("SMATMUL_OS not found in ballISA"))

  private val wsFunct7 = b.ballDomain.ballISA
    .find(_.mnemonic == "SMATMUL_WS")
    .map(_.funct7)
    .getOrElse(throw new IllegalArgumentException("SMATMUL_WS not found in ballISA"))

  require(
    b.memDomain.bankEntries % tile == 0,
    "SystolicArrayCtrl expects bankEntries to be an integer number of 16-row A/B tiles"
  )
  require(
    3 * addrWidth <= iterLen,
    s"SystolicArrayCtrl: iter ($iterLen bits) cannot hold 3 bases of $addrWidth bits each"
  )

  @public
  val io = IO(new Bundle {
    val cmdReq            = Flipped(Decoupled(new BallRsIssue(b)))
    val cmdResp_o         = Decoupled(new BallRsComplete(b))
    val ctrl_ld_o         = Decoupled(new SystolicCtrlLoadReq(b))
    val store_ctrl_resp_o = Decoupled(new SystolicStoreCtrlResp(b))
    val store_done_i      = Input(Bool())
    val busy_o            = Output(Bool())
  })

  private def fitTo(x: UInt, width: Int): UInt =
    if (x.getWidth >= width) x(width - 1, 0) else x.pad(width)

  private def ceilDiv16(x: UInt): UInt =
    ((x.pad(countWidth) +& (tile - 1).U) >> log2Ceil(tile)).pad(countWidth)

  private def abGroup(base: UInt, rowOffset: UInt): UInt = {
    val linear = base.pad(countWidth) + rowOffset.pad(countWidth)
    fitTo(linear / abRows.U, groupWidth)
  }

  private def abRow(base: UInt, rowOffset: UInt): UInt = {
    val linear = base.pad(countWidth) + rowOffset.pad(countWidth)
    fitTo(linear % abRows.U, addrWidth)
  }

  private def cGroupBase(base: UInt, blockCount: UInt): UInt = {
    val linearBlock = base.pad(countWidth) + blockCount.pad(countWidth)
    val groupPage   = linearBlock / b.memDomain.bankEntries.U
    fitTo(groupPage << log2Ceil(SystolicArrayConst.StoreWritePorts), groupWidth)
  }

  private def cRowAddr(base: UInt, blockCount: UInt): UInt = {
    val linearBlock = base.pad(countWidth) + blockCount.pad(countWidth)
    fitTo(linearBlock % b.memDomain.bankEntries.U, addrWidth)
  }

  private def validTileExtent(dim: UInt, tileIndex: UInt): UInt = {
    val remaining = dim.pad(countWidth) - (tileIndex << log2Ceil(tile))
    Mux(remaining >= tile.U, tile.U(5.W), fitTo(remaining, 5))
  }

  val issueQ = Module(new Queue(new BallRsIssue(b), entries = 4))
  issueQ.io.enq.valid := io.cmdReq.valid
  issueQ.io.enq.bits  := io.cmdReq.bits
  io.cmdReq.ready     := issueQ.io.enq.ready

  val retireQ                = Module(new Queue(new SystolicRetireMeta(b), entries = 4))
  val storeMetaQ             = Module(new Queue(new SystolicStoreMeta(b), entries = 4))
  val completeQ              = Module(new Queue(new BallRsComplete(b), entries = 4))
  val completionReservations = RegInit(0.U(log2Ceil(4 + 1).W))

  val issueActive = RegInit(false.B)
  val task        = RegInit(0.U.asTypeOf(new SystolicTask(b)))

  val mTiles = RegInit(0.U(countWidth.W))
  val kTiles = RegInit(0.U(countWidth.W))
  val mt     = RegInit(0.U(countWidth.W))
  val kt     = RegInit(0.U(countWidth.W))

  val wsBatchBase         = RegInit(0.U(countWidth.W))
  val wsLoadPeB           = RegInit(false.B)
  val wsWeightGeneration  = RegInit(false.B)
  val wsPrefetchedWeights = RegInit(false.B)

  val storeDoneMt         = RegInit(0.U(countWidth.W))
  val storeDoneRowIdx     = RegInit(0.U(5.W))
  val storeMetaMt         = RegInit(0.U(countWidth.W))
  val storeMetaRowIdx     = RegInit(0.U(5.W))
  val storeMetaBlockCount = RegInit(0.U(countWidth.W))
  val storeOutstanding    = RegInit(0.U(5.W))

  val cmd         = issueQ.io.deq.bits.cmd
  val iter        = cmd.iter
  val rs2         = cmd.rs2
  val issueFunct7 = cmd.funct7
  val issueWs     = issueFunct7 === wsFunct7.U(7.W)

  val issueM = rs2(11, 0)
  val issueN = rs2(23, 12)
  val issueK = rs2(35, 24)

  val nextTask = WireDefault(0.U.asTypeOf(new SystolicTask(b)))
  nextTask.m          := issueM
  nextTask.n          := issueN
  nextTask.k          := issueK
  nextTask.is_ws      := issueWs
  nextTask.op1_bank   := cmd.op1_bank
  nextTask.op2_bank   := cmd.op2_bank
  nextTask.wr_bank    := cmd.wr_bank
  nextTask.op1_base   := fitTo(iter(addrWidth - 1, 0), addrWidth)
  nextTask.op2_base   := fitTo(iter(2 * addrWidth - 1, addrWidth), addrWidth)
  nextTask.wr_base    := fitTo(iter(3 * addrWidth - 1, 2 * addrWidth), addrWidth)
  nextTask.rob_id     := issueQ.io.deq.bits.rob_id
  nextTask.is_sub     := issueQ.io.deq.bits.is_sub
  nextTask.sub_rob_id := issueQ.io.deq.bits.sub_rob_id
  if (3 * addrWidth < iterLen) {
    val iterUnused = iter(iterLen - 1, 3 * addrWidth)
    assert(
      !issueQ.io.deq.valid || iterUnused === 0.U,
      "SystolicArrayCtrl: unused iter bits must be 0"
    )
  }

  val currentKTileKind = Mux(
    kTiles === 1.U,
    SystolicKTileKind.DIRECT,
    Mux(
      kt === 0.U,
      SystolicKTileKind.FIRST,
      Mux(
        kt === (kTiles - 1.U),
        SystolicKTileKind.LAST,
        SystolicKTileKind.MIDDLE
      )
    )
  )

  val currentValidM = validTileExtent(task.m, mt)
  val currentValidN = task.n(4, 0)
  val currentValidK = validTileExtent(task.k, kt)

  val aRowOffset =
    (mt.pad(countWidth) * kTiles.pad(countWidth) + kt.pad(
      countWidth
    )) << log2Ceil(tile)

  val bRowOffset = kt.pad(countWidth) << log2Ceil(tile)

  val wsBatchLimit = Mux(
    wsBatchBase + wsReuseTiles.U < mTiles,
    wsBatchBase + wsReuseTiles.U,
    mTiles
  )

  val wsReuseCount       = wsBatchLimit - wsBatchBase
  val wsHasNextWeight    =
    kt + 1.U < kTiles || wsBatchBase + wsReuseTiles.U < mTiles
  val wsPrefetchB        = task.is_ws && !wsLoadPeB && wsReuseCount >= 3.U &&
    mt + 2.U === wsBatchLimit && wsHasNextWeight
  val prefetchKt         = Mux(kt + 1.U < kTiles, kt + 1.U, 0.U)
  val prefetchValidK     = validTileExtent(task.k, prefetchKt)
  val prefetchBRowOffset = prefetchKt.pad(countWidth) << log2Ceil(tile)
  val selectedBRowOffset = Mux(wsPrefetchB, prefetchBRowOffset, bRowOffset)

  val currentKind = Mux(
    task.is_ws,
    Mux(
      wsLoadPeB,
      SystolicCtrlLoadReqKind.READ_A_B_PE,
      Mux(
        wsPrefetchB,
        SystolicCtrlLoadReqKind.READ_A_B_BUF,
        SystolicCtrlLoadReqKind.READ_A_ONLY
      )
    ),
    SystolicCtrlLoadReqKind.READ_AB
  )

  val storeTask          = retireQ.io.deq.bits
  val storeMTiles        = ceilDiv16(storeTask.m)
  val storeMetaTask      = storeMetaQ.io.deq.bits
  val storeMetaMTiles    = ceilDiv16(storeMetaTask.m)
  val storeMetaValidCols = storeMetaTask.n(4, 0)
  val storeMetaValidRows = validTileExtent(storeMetaTask.m, storeMetaMt)

  val nextStoreResp = WireDefault(0.U.asTypeOf(new SystolicStoreCtrlResp(b)))
  nextStoreResp.row_valid_elems := storeMetaValidCols
  nextStoreResp.rob_id          := storeMetaTask.rob_id
  nextStoreResp.wr_bank         := storeMetaTask.wr_bank
  nextStoreResp.wr_group_base   := cGroupBase(
    storeMetaTask.wr_base,
    storeMetaBlockCount
  )
  nextStoreResp.wr_row_addr     := cRowAddr(
    storeMetaTask.wr_base,
    storeMetaBlockCount
  )

  val storeDoneValidRows = validTileExtent(storeTask.m, storeDoneMt)
  val storeDoneLastTile  = storeDoneMt === (storeMTiles - 1.U)
  val storeDoneLastRow   = storeDoneRowIdx + 1.U >= storeDoneValidRows
  val storeDoneTaskDone  = storeDoneLastTile && storeDoneLastRow

  val storeMetaLastTile = storeMetaMt === (storeMetaMTiles - 1.U)
  val storeMetaLastRow  = storeMetaRowIdx + 1.U >= storeMetaValidRows
  val storeMetaTaskDone = storeMetaLastTile && storeMetaLastRow

  val issueLastTile     =
    issueActive && mt === (mTiles - 1.U) && kt === (kTiles - 1.U)
  val queuedTaskReady   =
    issueQ.io.deq.valid && retireQ.io.enq.ready && storeMetaQ.io.enq.ready
  val canStartIssueTask = !issueActive && queuedTaskReady
  val replaceIssueTask  = io.ctrl_ld_o.fire && issueLastTile && queuedTaskReady
  val acceptIssueTask   = canStartIssueTask || replaceIssueTask

  issueQ.io.deq.ready            := acceptIssueTask
  retireQ.io.enq.valid           := acceptIssueTask
  retireQ.io.enq.bits.m          := nextTask.m
  retireQ.io.enq.bits.n          := nextTask.n
  retireQ.io.enq.bits.rob_id     := nextTask.rob_id
  retireQ.io.enq.bits.is_sub     := nextTask.is_sub
  retireQ.io.enq.bits.sub_rob_id := nextTask.sub_rob_id
  storeMetaQ.io.enq.valid        := acceptIssueTask
  storeMetaQ.io.enq.bits.m       := nextTask.m
  storeMetaQ.io.enq.bits.n       := nextTask.n
  storeMetaQ.io.enq.bits.wr_bank := nextTask.wr_bank
  storeMetaQ.io.enq.bits.wr_base := nextTask.wr_base
  storeMetaQ.io.enq.bits.rob_id  := nextTask.rob_id

  io.ctrl_ld_o.valid                  := issueActive
  io.ctrl_ld_o.bits.req_kind          := currentKind
  io.ctrl_ld_o.bits.k_tile_kind       := currentKTileKind
  io.ctrl_ld_o.bits.acc_slot          := Mux(
    task.is_ws,
    fitTo(mt - wsBatchBase, log2Ceil(wsReuseTiles)),
    0.U
  )
  io.ctrl_ld_o.bits.valid_m           := currentValidM
  io.ctrl_ld_o.bits.valid_n           := currentValidN
  io.ctrl_ld_o.bits.valid_k           := currentValidK
  io.ctrl_ld_o.bits.b_valid_n         := currentValidN
  io.ctrl_ld_o.bits.b_valid_k         := Mux(wsPrefetchB, prefetchValidK, currentValidK)
  io.ctrl_ld_o.bits.weight_generation := wsWeightGeneration
  io.ctrl_ld_o.bits.op1_bank          := task.op1_bank
  io.ctrl_ld_o.bits.op1_group         := abGroup(task.op1_base, aRowOffset)
  io.ctrl_ld_o.bits.op1_row_base      := abRow(task.op1_base, aRowOffset)
  io.ctrl_ld_o.bits.op2_bank          := task.op2_bank
  io.ctrl_ld_o.bits.op2_group         := abGroup(task.op2_base, selectedBRowOffset)
  io.ctrl_ld_o.bits.op2_row_base      := abRow(task.op2_base, selectedBRowOffset)

  io.cmdResp_o.valid               := completeQ.io.deq.valid
  io.cmdResp_o.bits                := completeQ.io.deq.bits
  completeQ.io.deq.ready           := io.cmdResp_o.ready
  completeQ.io.enq.valid           := false.B
  completeQ.io.enq.bits.rob_id     := storeTask.rob_id
  completeQ.io.enq.bits.is_sub     := storeTask.is_sub
  completeQ.io.enq.bits.sub_rob_id := storeTask.sub_rob_id
  retireQ.io.deq.ready             := false.B

  io.busy_o := issueActive || issueQ.io.deq.valid || retireQ.io.deq.valid ||
    completeQ.io.deq.valid || storeOutstanding =/= 0.U
  val completionSlotAvailable =
    completionReservations < 4.U || completeQ.io.deq.fire
  val storeCanEnqueue         = storeMetaQ.io.deq.valid &&
    (!storeMetaTaskDone || completionSlotAvailable)
  io.store_ctrl_resp_o.valid := storeCanEnqueue
  io.store_ctrl_resp_o.bits  := nextStoreResp

  val storeMetaIssue    = io.store_ctrl_resp_o.fire
  val reserveCompletion = storeMetaIssue && storeMetaTaskDone
  storeMetaQ.io.deq.ready := storeMetaIssue && storeMetaTaskDone
  switch(Cat(reserveCompletion, completeQ.io.deq.fire)) {
    is("b10".U)(completionReservations := completionReservations + 1.U)
    is("b01".U)(completionReservations := completionReservations - 1.U)
  }
  when(storeMetaIssue) {
    when(storeMetaTaskDone) {
      storeMetaMt         := 0.U
      storeMetaRowIdx     := 0.U
      storeMetaBlockCount := 0.U
    }.otherwise {
      storeMetaBlockCount := storeMetaBlockCount + 1.U
      storeMetaRowIdx     := storeMetaRowIdx + 1.U
      when(storeMetaLastRow) {
        storeMetaRowIdx := 0.U
        storeMetaMt     := storeMetaMt + 1.U
      }
    }
  }
  assert(
    completionReservations <= 4.U,
    "SystolicArrayCtrl: completion reservation overflow"
  )

  switch(Cat(io.store_ctrl_resp_o.fire, io.store_done_i)) {
    is("b10".U)(storeOutstanding := storeOutstanding + 1.U)
    is("b01".U)(storeOutstanding := storeOutstanding - 1.U)
  }

  when(io.store_done_i) {
    assert(
      storeOutstanding =/= 0.U,
      "SystolicArrayCtrl: store_done without an issued descriptor"
    )

    when(storeDoneTaskDone) {
      assert(
        completeQ.io.enq.ready,
        "SystolicArrayCtrl: final store row completed without completion queue space"
      )
      completeQ.io.enq.valid := true.B
      retireQ.io.deq.ready   := completeQ.io.enq.ready
      storeDoneMt            := 0.U
      storeDoneRowIdx        := 0.U
    }.otherwise {
      storeDoneRowIdx := storeDoneRowIdx + 1.U
      when(storeDoneLastRow) {
        storeDoneRowIdx := 0.U
        storeDoneMt     := storeDoneMt + 1.U
      }
    }
  }
  assert(
    storeOutstanding <= 8.U,
    "SystolicArrayCtrl: Store completion pipeline overflow"
  )

  when(io.ctrl_ld_o.fire) {
    when(kTiles === 1.U) {
      assert(currentKTileKind === SystolicKTileKind.DIRECT)
    }.elsewhen(kt === 0.U) {
      assert(currentKTileKind === SystolicKTileKind.FIRST)
    }.elsewhen(kt === (kTiles - 1.U)) {
      assert(currentKTileKind === SystolicKTileKind.LAST)
    }.otherwise {
      assert(currentKTileKind === SystolicKTileKind.MIDDLE)
    }

    when(issueLastTile) {
      when(!replaceIssueTask) {
        issueActive := false.B
      }
    }.otherwise {
      when(task.is_ws) {
        advanceIssueWsLoad()
      }.otherwise {
        advanceIssueOsTile()
      }
    }
  }

  private def advanceIssueOsTile(): Unit = {
    when(kt + 1.U < kTiles) {
      kt := kt + 1.U
    }.otherwise {
      kt := 0.U
      when(mt + 1.U < mTiles) {
        mt := mt + 1.U
      }.otherwise {
        assert(false.B, "SystolicArrayCtrl: advanced past final OS tile")
      }
    }
  }

  private def advanceIssueWsLoad(): Unit = {
    when(mt + 1.U < wsBatchLimit) {
      mt        := mt + 1.U
      wsLoadPeB := false.B
      when(wsPrefetchB) {
        wsPrefetchedWeights := true.B
      }
    }.otherwise {
      mt := wsBatchBase
      when(kt + 1.U < kTiles) {
        kt                  := kt + 1.U
        wsLoadPeB           := !wsPrefetchedWeights
        wsWeightGeneration  := !wsWeightGeneration
        wsPrefetchedWeights := false.B
      }.otherwise {
        kt := 0.U
        when(wsBatchBase + wsReuseTiles.U < mTiles) {
          wsBatchBase         := wsBatchBase + wsReuseTiles.U
          mt                  := wsBatchBase + wsReuseTiles.U
          wsLoadPeB           := !wsPrefetchedWeights
          wsWeightGeneration  := !wsWeightGeneration
          wsPrefetchedWeights := false.B
        }.otherwise {
          assert(false.B, "SystolicArrayCtrl: advanced past final WS tile")
        }
      }
    }
  }

  private def startIssueTask(): Unit = {
    task                := nextTask
    mTiles              := ceilDiv16(issueM)
    kTiles              := ceilDiv16(issueK)
    mt                  := 0.U
    kt                  := 0.U
    wsBatchBase         := 0.U
    wsLoadPeB           := issueWs
    wsWeightGeneration  := false.B
    wsPrefetchedWeights := false.B
    issueActive         := true.B

    assert(
      issueFunct7 === osFunct7.U(7.W) || issueFunct7 === wsFunct7.U(7.W),
      "SystolicArrayCtrl: funct7 must be SMATMUL_OS or SMATMUL_WS"
    )
    assert(
      issueM =/= 0.U && issueN =/= 0.U && issueK =/= 0.U,
      "SystolicArrayCtrl: rows/cols/k must be non-zero"
    )
    assert(issueN <= tile.U(12.W), "SystolicArrayCtrl: cols must be 1..16")
    assert(rs2(63, 36) === 0.U(28.W), "SystolicArrayCtrl: rs2[63:36] must be 0")
  }

  when(acceptIssueTask) {
    startIssueTask()
  }
}
