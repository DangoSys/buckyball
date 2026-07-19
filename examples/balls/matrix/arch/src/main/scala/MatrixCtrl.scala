package examples.balls.matrix

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.balldomain.rs.{BallRsComplete, BallRsIssue}
import framework.top.GlobalConfig

object MatrixConst {
  val Tile              = 16
  val OpElemBits        = 8
  val AccElemBits       = 32
  val OpRowBits         = Tile * OpElemBits
  val ResultRowBits     = Tile * AccElemBits
  val StoreWritePorts   = 4
  val StorePortElemCount = Tile / StoreWritePorts
  val WsReuseTiles       = 8
}

object MatrixCtrlLoadReqKind {
  val READ_AB       = 0.U(2.W)
  val READ_A_ONLY   = 1.U(2.W)
  val READ_A_B_PE   = 2.U(2.W)
}

object MatrixKTileKind {
  val DIRECT = 0.U(2.W)
  val FIRST  = 1.U(2.W)
  val MIDDLE = 2.U(2.W)
  val LAST   = 3.U(2.W)
}

class MatrixCtrlLoadReq(b: GlobalConfig) extends Bundle {
  val req_kind     = UInt(2.W)
  val k_tile_kind  = UInt(2.W)
  val acc_slot     = UInt(log2Ceil(MatrixConst.WsReuseTiles).W)
  val valid_m      = UInt(5.W)
  val valid_n      = UInt(5.W)
  val valid_k      = UInt(5.W)
  val row_count    = UInt(5.W)
  val op1_bank     = UInt(log2Up(b.memDomain.bankNum).W)
  val op1_group    = UInt(log2Up(b.memDomain.bankNum).W)
  val op1_row_base = UInt(log2Up(b.memDomain.bankEntries).W)
  val op2_bank     = UInt(log2Up(b.memDomain.bankNum).W)
  val op2_group    = UInt(log2Up(b.memDomain.bankNum).W)
  val op2_row_base = UInt(log2Up(b.memDomain.bankEntries).W)
}

class MatrixResultRow extends Bundle {
  val data = UInt(MatrixConst.ResultRowBits.W)
}

class MatrixStoreCtrlResp(b: GlobalConfig) extends Bundle {
  val row_write_valid = Bool()
  val row_valid_elems = UInt(5.W)
  val wr_bank         = UInt(log2Up(b.memDomain.bankNum).W)
  val wr_group_base   = UInt(log2Up(b.memDomain.bankNum).W)
  val wr_row_addr     = UInt(log2Up(b.memDomain.bankEntries).W)
}

class MatrixStoreWriteReq(b: GlobalConfig) extends Bundle {
  val wr_bank       = UInt(log2Up(b.memDomain.bankNum).W)
  val wr_group_base = UInt(log2Up(b.memDomain.bankNum).W)
  val wr_row_addr   = UInt(log2Up(b.memDomain.bankEntries).W)
  val valid_elems   = UInt(5.W)
  val beat_count    = UInt(3.W)
  val data          = UInt(MatrixConst.ResultRowBits.W)
}

class MatrixTask(b: GlobalConfig) extends Bundle {
  val mode       = Bool()
  val m          = UInt(12.W)
  val n          = UInt(12.W)
  val k          = UInt(12.W)
  val op1_bank   = UInt(log2Up(b.memDomain.bankNum).W)
  val op2_bank   = UInt(log2Up(b.memDomain.bankNum).W)
  val wr_bank    = UInt(log2Up(b.memDomain.bankNum).W)
  val op1_base   = UInt(log2Up(b.memDomain.bankEntries).W)
  val op2_base   = UInt(log2Up(b.memDomain.bankEntries).W)
  val wr_base    = UInt(log2Up(b.memDomain.bankEntries).W)
  val rob_id     = UInt(log2Up(b.frontend.rob_entries).W)
  val is_sub     = Bool()
  val sub_rob_id = UInt(log2Up(b.frontend.sub_rob_depth * 4).W)
}

@instantiable
class MatrixCtrl(val b: GlobalConfig) extends Module {
  private val tile       = MatrixConst.Tile
  private val abRows     = b.memDomain.bankEntries
  private val bankWidth  = log2Up(b.memDomain.bankNum)
  private val groupWidth = log2Up(b.memDomain.bankNum)
  private val addrWidth  = log2Up(b.memDomain.bankEntries)
  private val countWidth = 32

  require(
    b.memDomain.bankEntries % tile == 0,
    "MatrixCtrl expects bankEntries to be an integer number of 16-row A/B tiles"
  )

  @public
  val io = IO(new Bundle {
    val cmdReq    = Flipped(Decoupled(new BallRsIssue(b)))
    val cmdResp_o = Decoupled(new BallRsComplete(b))

    val ctrl_ld_o = Decoupled(new MatrixCtrlLoadReq(b))

    val store_req_i       = Input(Bool())
    val store_ctrl_resp_o = Decoupled(new MatrixStoreCtrlResp(b))
    val store_done_i      = Input(Bool())

    val busy_o = Output(Bool())
    val active_rob_id_o = Output(UInt(log2Up(b.frontend.rob_entries).W))
  })

  private def fitTo(x: UInt, width: Int): UInt =
    if (x.getWidth >= width) x(width - 1, 0) else x.pad(width)

  private def ceilDiv16(x: UInt): UInt =
    ((x.pad(countWidth) +& (tile - 1).U) >> log2Ceil(tile)).pad(countWidth)

  private def abGroup(base: UInt, tileLinear: UInt): UInt = {
    val linear = base.pad(countWidth) + (tileLinear.pad(countWidth) << log2Ceil(tile))
    fitTo(linear / abRows.U, groupWidth)
  }

  private def abRow(base: UInt, tileLinear: UInt): UInt = {
    val linear = base.pad(countWidth) + (tileLinear.pad(countWidth) << log2Ceil(tile))
    fitTo(linear % abRows.U, addrWidth)
  }

  private def cGroupBase(base: UInt, beatCount: UInt): UInt = {
    val linearBeat   = base.pad(countWidth) + beatCount.pad(countWidth)
    val logicalGroup = linearBeat / b.memDomain.bankEntries.U
    fitTo(logicalGroup, groupWidth)
  }

  private def cRowAddr(base: UInt, beatCount: UInt): UInt = {
    val linearBeat = base.pad(countWidth) + beatCount.pad(countWidth)
    fitTo(linearBeat % b.memDomain.bankEntries.U, addrWidth)
  }

  private def cRowBeats(validCols: UInt): UInt =
    fitTo((validCols.pad(countWidth) + 3.U) >> 2, 3)

  private def validTileExtent(dim: UInt, tileIndex: UInt): UInt = {
    val remaining = dim.pad(countWidth) - (tileIndex << log2Ceil(tile))
    Mux(remaining >= tile.U, tile.U(5.W), fitTo(remaining, 5))
  }

  val issueQ = Module(new Queue(new BallRsIssue(b), entries = 4))
  issueQ.io.enq.valid := io.cmdReq.valid
  issueQ.io.enq.bits  := io.cmdReq.bits
  io.cmdReq.ready     := issueQ.io.enq.ready

  val sIdle :: sRun :: sResp :: Nil = Enum(3)
  val state = RegInit(sIdle)

  val task = RegInit(0.U.asTypeOf(new MatrixTask(b)))

  val mTiles = RegInit(0.U(countWidth.W))
  val nTiles = RegInit(0.U(countWidth.W))
  val kTiles = RegInit(0.U(countWidth.W))

  val mt = RegInit(0.U(countWidth.W))
  val nt = RegInit(0.U(countWidth.W))
  val kt = RegInit(0.U(countWidth.W))
  val wsBatchBase = RegInit(0.U(countWidth.W))

  val wsLoadPeB = RegInit(false.B)
  val issueDone = RegInit(false.B)
  val storeDoneAll = RegInit(false.B)

  val storeMt       = RegInit(0.U(countWidth.W))
  val storeNt       = RegInit(0.U(countWidth.W))
  val storeRowIdx   = RegInit(0.U(5.W))

  val cOutputBeatCount = RegInit(0.U(countWidth.W))

  val storeRespValid = RegInit(false.B)
  val storeRespBits  = RegInit(0.U.asTypeOf(new MatrixStoreCtrlResp(b)))
  val storeRespRowValidReg = RegInit(false.B)
  val storeRespAccepted = RegInit(false.B)

  val rs1 = issueQ.io.deq.bits.cmd.rs1
  val rs2 = issueQ.io.deq.bits.cmd.rs2

  val issueM    = rs2(11, 0)
  val issueN    = rs2(23, 12)
  val issueK    = rs2(35, 24)
  val issueMode = rs2(36)

  val nextTask = WireDefault(0.U.asTypeOf(new MatrixTask(b)))
  nextTask.mode       := issueMode
  nextTask.m          := issueM
  nextTask.n          := issueN
  nextTask.k          := issueK
  nextTask.op1_bank   := fitTo(rs1(9, 0), bankWidth)
  nextTask.op2_bank   := fitTo(rs1(19, 10), bankWidth)
  nextTask.wr_bank    := fitTo(rs1(29, 20), bankWidth)
  nextTask.op1_base   := fitTo(rs1(36, 30), addrWidth)
  nextTask.op2_base   := fitTo(rs1(43, 37), addrWidth)
  nextTask.wr_base    := fitTo(rs1(50, 44), addrWidth)
  nextTask.rob_id     := issueQ.io.deq.bits.rob_id
  nextTask.is_sub     := issueQ.io.deq.bits.is_sub
  nextTask.sub_rob_id := issueQ.io.deq.bits.sub_rob_id

  val currentKind = Wire(UInt(2.W))
  currentKind := Mux(
    task.mode,
    Mux(wsLoadPeB, MatrixCtrlLoadReqKind.READ_A_B_PE, MatrixCtrlLoadReqKind.READ_A_ONLY),
    MatrixCtrlLoadReqKind.READ_AB
  )
  val currentKTileKind = Mux(
    kTiles === 1.U,
    MatrixKTileKind.DIRECT,
    Mux(kt === 0.U, MatrixKTileKind.FIRST,
      Mux(kt === (kTiles - 1.U), MatrixKTileKind.LAST, MatrixKTileKind.MIDDLE))
  )
  val aTileLinear = mt * kTiles + kt
  val bTileLinear = nt * kTiles + kt
  val currentValidM = validTileExtent(task.m, mt)
  val currentValidN = validTileExtent(task.n, nt)
  val currentValidK = validTileExtent(task.k, kt)
  val currentInputRows = MuxLookup(currentKind, currentValidM)(Seq(
    MatrixCtrlLoadReqKind.READ_AB -> Mux(currentValidM >= currentValidK, currentValidM, currentValidK),
    MatrixCtrlLoadReqKind.READ_A_ONLY -> currentValidM,
    MatrixCtrlLoadReqKind.READ_A_B_PE -> tile.U(5.W)
  ))
  val wsBatchLimit = Mux(
    wsBatchBase + MatrixConst.WsReuseTiles.U < mTiles,
    wsBatchBase + MatrixConst.WsReuseTiles.U,
    mTiles)

  val remainingCols = task.n.pad(countWidth) - (storeNt << log2Ceil(tile))
  val validCols = Mux(remainingCols >= tile.U, tile.U(5.W), fitTo(remainingCols, 5))
  val validRows = validTileExtent(task.m, storeMt)
  val validRowBeats = cRowBeats(validCols)
  val globalRow = (storeMt << log2Ceil(tile)) + storeRowIdx
  val rowInRange = globalRow < task.m.pad(countWidth)
  val fullRowBeats = (task.n.pad(countWidth) + 3.U) >> 2
  val wsCompactBeatBase =
    (storeMt.pad(countWidth) << log2Ceil(tile)) * fullRowBeats +
      (storeNt.pad(countWidth) << log2Ceil(tile / 4)) * validRows.pad(countWidth) +
      storeRowIdx.pad(countWidth) * validRowBeats.pad(countWidth)
  val storeBeatBase = Mux(task.mode, wsCompactBeatBase, cOutputBeatCount)

  val nextStoreResp = WireDefault(0.U.asTypeOf(new MatrixStoreCtrlResp(b)))
  nextStoreResp.row_write_valid := rowInRange
  nextStoreResp.row_valid_elems := Mux(rowInRange, validCols, 0.U)
  nextStoreResp.wr_bank         := task.wr_bank
  nextStoreResp.wr_group_base   := cGroupBase(task.wr_base, storeBeatBase)
  nextStoreResp.wr_row_addr     := cRowAddr(task.wr_base, storeBeatBase)

  val storeLastTile = storeMt === (mTiles - 1.U) && storeNt === (nTiles - 1.U)

  io.ctrl_ld_o.valid := state === sRun && !issueDone
  io.ctrl_ld_o.bits.req_kind     := currentKind
  io.ctrl_ld_o.bits.k_tile_kind  := currentKTileKind
  io.ctrl_ld_o.bits.acc_slot     := Mux(
    task.mode,
    fitTo(mt - wsBatchBase, log2Ceil(MatrixConst.WsReuseTiles)),
    0.U)
  io.ctrl_ld_o.bits.valid_m      := currentValidM
  io.ctrl_ld_o.bits.valid_n      := currentValidN
  io.ctrl_ld_o.bits.valid_k      := currentValidK
  io.ctrl_ld_o.bits.row_count    := currentInputRows
  io.ctrl_ld_o.bits.op1_bank     := task.op1_bank
  io.ctrl_ld_o.bits.op1_group    := abGroup(task.op1_base, aTileLinear)
  io.ctrl_ld_o.bits.op1_row_base := abRow(task.op1_base, aTileLinear)
  io.ctrl_ld_o.bits.op2_bank     := task.op2_bank
  io.ctrl_ld_o.bits.op2_group    := abGroup(task.op2_base, bTileLinear)
  io.ctrl_ld_o.bits.op2_row_base := abRow(task.op2_base, bTileLinear)

  io.store_ctrl_resp_o.valid := storeRespValid
  io.store_ctrl_resp_o.bits  := storeRespBits

  val taskComplete = state === sRun && issueDone && storeDoneAll

  io.cmdResp_o.valid           := state === sResp || taskComplete
  io.cmdResp_o.bits.rob_id     := task.rob_id
  io.cmdResp_o.bits.is_sub     := task.is_sub
  io.cmdResp_o.bits.sub_rob_id := task.sub_rob_id

  io.busy_o := state =/= sIdle || issueQ.io.deq.valid || io.cmdResp_o.valid
  io.active_rob_id_o := task.rob_id

  issueQ.io.deq.ready := state === sIdle || io.cmdResp_o.fire

  when(io.store_req_i) {
    assert(state === sRun, "MatrixCtrl: store_req arrived outside task execution")
    assert(!storeRespValid, "MatrixCtrl: store_req arrived while previous store response is still pending")
    assert(!storeRespAccepted, "MatrixCtrl: store_req arrived before previous store row completed")
    storeRespBits  := nextStoreResp
    storeRespValid := true.B
  }

  when(io.store_ctrl_resp_o.fire) {
    storeRespValid         := false.B
    storeRespAccepted      := true.B
    storeRespRowValidReg   := storeRespBits.row_write_valid
  }

  when(io.store_done_i) {
    assert(storeRespAccepted, "MatrixCtrl: store_done without an accepted store response")
    storeRespAccepted := false.B

    when(storeRespRowValidReg) {
      cOutputBeatCount := cOutputBeatCount + validRowBeats
    }

    storeRowIdx := storeRowIdx + 1.U

    when(storeRowIdx + 1.U >= validRows) {
      storeRowIdx := 0.U

      when(storeLastTile) {
        storeDoneAll := true.B
      }.elsewhen(task.mode) {
        when(storeMt + 1.U < mTiles) {
          storeMt := storeMt + 1.U
        }.otherwise {
          storeMt := 0.U
          when(storeNt + 1.U < nTiles) {
            storeNt := storeNt + 1.U
          }
        }
      }.otherwise {
        when(storeNt + 1.U < nTiles) {
          storeNt := storeNt + 1.U
        }.otherwise {
          storeNt := 0.U
          when(storeMt + 1.U < mTiles) {
            storeMt := storeMt + 1.U
          }
        }
      }
    }
  }

  when(io.ctrl_ld_o.fire) {
    when(kTiles === 1.U) {
      assert(currentKTileKind === MatrixKTileKind.DIRECT)
    }.elsewhen(kt === 0.U) {
      assert(currentKTileKind === MatrixKTileKind.FIRST)
    }.elsewhen(kt === (kTiles - 1.U)) {
      assert(currentKTileKind === MatrixKTileKind.LAST)
    }.otherwise {
      assert(currentKTileKind === MatrixKTileKind.MIDDLE)
    }
  }

  private def advanceIssueOsTile(): Unit = {
    when(kt + 1.U < kTiles) {
      kt := kt + 1.U
    }.otherwise {
      kt := 0.U
      when(nt + 1.U < nTiles) {
        nt := nt + 1.U
      }.otherwise {
        nt := 0.U
        when(mt + 1.U < mTiles) {
          mt := mt + 1.U
        }.otherwise {
          issueDone := true.B
        }
      }
    }
  }

  private def advanceIssueWsLoad(): Unit = {
    when(mt + 1.U < wsBatchLimit) {
      mt := mt + 1.U
      wsLoadPeB := false.B
    }.otherwise {
      mt := wsBatchBase
      when(kt + 1.U < kTiles) {
        kt := kt + 1.U
        wsLoadPeB := true.B
      }.otherwise {
        kt := 0.U
        when(wsBatchBase + MatrixConst.WsReuseTiles.U < mTiles) {
          wsBatchBase := wsBatchBase + MatrixConst.WsReuseTiles.U
          mt := wsBatchBase + MatrixConst.WsReuseTiles.U
          wsLoadPeB := true.B
        }.otherwise {
          wsBatchBase := 0.U
          mt := 0.U
          when(nt + 1.U < nTiles) {
            nt := nt + 1.U
            wsLoadPeB := true.B
          }.otherwise {
            issueDone := true.B
          }
        }
      }
    }
  }

  private def startNextTask(): Unit = {
    task   := nextTask
    mTiles := ceilDiv16(issueM)
    nTiles := ceilDiv16(issueN)
    kTiles := ceilDiv16(issueK)
    mt := 0.U
    nt := 0.U
    kt := 0.U
    wsBatchBase := 0.U
    storeMt := 0.U
    storeNt := 0.U
    storeRowIdx := 0.U
    cOutputBeatCount := 0.U
    issueDone := false.B
    storeDoneAll := false.B
    storeRespValid := false.B
    storeRespAccepted := false.B
    wsLoadPeB := issueMode
    state := sRun

    assert(issueM =/= 0.U && issueN =/= 0.U && issueK =/= 0.U, "MatrixCtrl: M/N/K must be non-zero")
  }

  switch(state) {
    is(sIdle) {
      when(issueQ.io.deq.valid) {
        startNextTask()
      }
    }

    is(sRun) {
      when(io.ctrl_ld_o.fire) {
        when(task.mode) {
          advanceIssueWsLoad()
        }.otherwise {
          advanceIssueOsTile()
        }
      }

      when(taskComplete) {
        when(io.cmdResp_o.fire) {
          when(issueQ.io.deq.valid) {
            startNextTask()
          }.otherwise {
            state := sIdle
          }
        }.otherwise {
          state := sResp
        }
      }
    }

    is(sResp) {
      when(io.cmdResp_o.fire) {
        when(issueQ.io.deq.valid) {
          startNextTask()
        }.otherwise {
          state := sIdle
        }
      }
    }
  }
}
