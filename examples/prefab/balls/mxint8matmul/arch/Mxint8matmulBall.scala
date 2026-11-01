package examples.balls.mxint8matmul

import framework.acc.balldomain.bbus.{BallFactory, BallPorts}
import buckyball.balls.mxint8matmul.{Params => ParametersProto}
import framework.acc.balldomain.configs.BallDomainParam
import framework.acc.frontend.configs.FrontendParam
import framework.acc.memdomain.configs.MemDomainParam
import framework.acc.balldomain.ball.blink.{BlinkIO, HasBlink}

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}

/**
 * MXINT8MM: C[M,N] (FP32) = A[M,K] . B[N,K]^T over packed MXINT8 rows. A rows x cols array of PEs walks the output
 * tiles row-major. Per tile: optionally read the old C (read port 0 on the pre-write bank), then per K block read the
 * two code rows and the exponent row of each A row (port 0) and B row (port 1), multiply `lanes` code pairs per PE per
 * cycle, convert the block sum to an FP32 term and add it into the accumulator; finally write the tile's C rows.
 */
@instantiable
class Mxint8matmulBall(
  fp: FrontendParam,
  mp: MemDomainParam,
  bp: BallDomainParam,
  id: Int)
    extends Module
    with HasBlink {
  val ball = bp.balls.find(_.id == id).get

  val params = ball.parameters.unpack(classOf[ParametersProto])
  require(params.hasRows && params.hasCols && params.hasLanes, "Mxint8matmulBall requires rows, cols and lanes")
  val rows   = params.getRows
  val cols   = params.getCols
  val lanes  = params.getLanes
  require(rows > 0 && cols > 0 && cols % 4 == 0, "Mxint8matmulBall needs a positive array with cols a multiple of 4")
  require(lanes > 0 && 32              % lanes == 0, "Mxint8matmulBall lanes must divide 32")
  require(mp.bankWidth == 128 && mp.bankMaskLen == 16, "Mxint8matmulBall requires 128-bit bank rows and byte masks")
  require(ball.inBW == 2 && ball.outBW == 1, "Mxint8matmulBall requires two reads and one write")
  require(fp.commandDataBits == 64, "Mxint8matmulBall decodes a 64-bit rs2")
  val steps = 32 / lanes

  @public
  val blink = IO(new BlinkIO(fp, mp, bp, id))

  val idle :: tile :: oldC :: load :: mac :: convert :: accumulate :: store :: complete :: Nil = Enum(9)
  val state                                                                                    = RegInit(idle)

  val robId   = Reg(UInt(log2Up(fp.rob_entries).W))
  val aBank   = Reg(UInt(fp.bank_id_len.W))
  val bBank   = Reg(UInt(fp.bank_id_len.W))
  val cBank   = Reg(UInt(fp.bank_id_len.W))
  val oldBank = Reg(UInt(fp.bank_id_len.W))
  val aGroup  = Reg(UInt(mp.groupIdWidth.W))
  val bGroup  = Reg(UInt(mp.groupIdWidth.W))
  val cGroup  = Reg(UInt(mp.groupIdWidth.W))
  val m       = Reg(UInt(10.W))
  val n       = Reg(UInt(10.W))
  val k       = Reg(UInt(fp.iter_len.W))
  val stride  = Reg(UInt(32.W))
  val bBase   = Reg(UInt(32.W))
  val fromOld = Reg(Bool())

  // Output tile origin and the bank rows of its first A row, first B row and first C row.
  val m0    = Reg(UInt(11.W))
  val n0    = Reg(UInt(11.W))
  val aTile = Reg(UInt(32.W))
  val bTile = Reg(UInt(32.W))
  val cTile = Reg(UInt(32.W))
  val block = Reg(UInt(fp.iter_len.W))
  val step  = Reg(UInt(log2Ceil(steps).max(1).W))

  val activeRows = Mux(m - m0 > rows.U, rows.U, m - m0)
  val activeCols = Mux(n - n0 > cols.U, cols.U, n - n0)
  val cGroups    = activeCols >> 2

  // Operand read sequence: per array row, code rows 0 and 1 and then the exponent row (part 2).
  val aIssueRow   = Reg(UInt(log2Up(rows + 1).W))
  val aIssuePart  = Reg(UInt(2.W))
  val aIssueAddr  = Reg(UInt(32.W))
  val aRespRow    = Reg(UInt(log2Up(rows + 1).W))
  val aRespPart   = Reg(UInt(2.W))
  val bIssueRow   = Reg(UInt(log2Up(cols + 1).W))
  val bIssuePart  = Reg(UInt(2.W))
  val bIssueAddr  = Reg(UInt(32.W))
  val bRespRow    = Reg(UInt(log2Up(cols + 1).W))
  val bRespPart   = Reg(UInt(2.W))
  // C sequence (old C reads and result writes): per array row, one bank row per four columns.
  val cIssueRow   = Reg(UInt(log2Up(rows + 1).W))
  val cIssueGroup = Reg(UInt(log2Up(cols / 4 + 1).W))
  val cIssueAddr  = Reg(UInt(32.W))
  val cRespRow    = Reg(UInt(log2Up(rows + 1).W))
  val cRespGroup  = Reg(UInt(log2Up(cols / 4 + 1).W))

  val aCodes = Reg(Vec(rows, Vec(2, UInt(128.W))))
  val aScale = Reg(Vec(rows, UInt(8.W)))
  val bCodes = Reg(Vec(cols, Vec(2, UInt(128.W))))
  val bScale = Reg(Vec(cols, UInt(8.W)))

  val scaleRow    = (k >> 4) + (block >> 4)
  val scaleOffset = Cat(block(3, 0), 0.U(3.W))

  val active = !reset.asBool
  blink.cmdReq.ready        := state === idle && active
  blink.cmdResp.valid       := state === complete && active
  blink.cmdResp.bits.rob_id := robId
  blink.status.idle         := state === idle
  blink.status.running      := state =/= idle && state =/= complete

  val readA = blink.bankRead(0)
  readA.rob_id           := robId
  readA.bank_id          := Mux(state === oldC, oldBank, aBank)
  readA.group_id         := Mux(state === oldC, cGroup, aGroup)
  readA.io.req.valid     := ((state === oldC && cIssueRow < activeRows) || (state === load && aIssueRow < activeRows)) &&
    active
  readA.io.req.bits.addr := Mux(
    state === oldC,
    cIssueAddr + cIssueGroup,
    aIssueAddr + Mux(aIssuePart === 2.U, scaleRow, Cat(block, aIssuePart(0)))
  )
  readA.io.resp.ready    := (state === oldC || state === load) && active

  val readB = blink.bankRead(1)
  readB.rob_id           := robId
  readB.bank_id          := bBank
  readB.group_id         := bGroup
  readB.io.req.valid     := state === load && bIssueRow < activeCols && active
  readB.io.req.bits.addr := bIssueAddr + Mux(bIssuePart === 2.U, scaleRow, Cat(block, bIssuePart(0)))
  readB.io.resp.ready    := state === load && active

  val pes = Seq.tabulate(rows, cols)((_, _) => Instantiate(new PE(lanes)))

  val write = blink.bankWrite(0)
  write.rob_id           := robId
  write.bank_id          := cBank
  write.group_id         := cGroup
  write.io.req.valid     := state === store && cIssueRow < activeRows && active
  write.io.req.bits.addr := cIssueAddr + cIssueGroup
  write.io.req.bits.data := VecInit(pes.map(row => VecInit(row.grouped(4).map(g => Cat(g.reverse.map(_.io.out))).toSeq)))(
    cIssueRow
  )(cIssueGroup)
  write.io.req.bits.mask := VecInit(Seq.fill(16)(true.B))
  write.io.resp.ready    := state === store && active

  val oldWords = (0 until 4).map(w => readA.io.resp.bits.data(32 * w + 31, 32 * w))
  // The `lanes` codes of the current step: bytes [step * lanes, (step + 1) * lanes) of the 32-code block.
  val aLanes   = aCodes.map(codes => Cat(codes(1), codes(0)).asTypeOf(Vec(steps, UInt((lanes * 8).W)))(step))
  val bLanes   = bCodes.map(codes => Cat(codes(1), codes(0)).asTypeOf(Vec(steps, UInt((lanes * 8).W)))(step))
  for {
    i <- 0 until rows
    j <- 0 until cols
  } {
    val pe = pes(i)(j)
    pe.io.active     := i.U < activeRows && j.U < activeCols
    pe.io.mac        := state === mac
    pe.io.first      := step === 0.U
    pe.io.a          := aLanes(i)
    pe.io.b          := bLanes(j)
    pe.io.convert    := state === convert
    pe.io.scale      := aScale(i).zext +& bScale(j).zext - 254.S
    pe.io.accumulate := state === accumulate
    // Each tile starts from +0.0 or from the old C element.
    pe.io.load       := (state === tile && !fromOld) ||
      (state === oldC && readA.io.resp.fire && cRespRow === i.U && cRespGroup === (j / 4).U)
    pe.io.init       := Mux(state === oldC, oldWords(j % 4), 0.U)
  }

  when(readA.io.req.fire) {
    when(state === oldC) {
      when(cIssueGroup === cGroups - 1.U) {
        cIssueGroup := 0.U
        cIssueRow   := cIssueRow + 1.U
        cIssueAddr  := cIssueAddr + (n >> 2)
      }.otherwise(cIssueGroup := cIssueGroup + 1.U)
    }.otherwise {
      when(aIssuePart === 2.U) {
        aIssuePart := 0.U
        aIssueRow  := aIssueRow + 1.U
        aIssueAddr := aIssueAddr + stride
      }.otherwise(aIssuePart := aIssuePart + 1.U)
    }
  }
  when(readA.io.resp.fire) {
    when(state === oldC) {
      oldWords.foreach(word => assert(word(30, 23) =/= 255.U, "MXINT8MM old C must be finite"))
      when(cRespGroup === cGroups - 1.U) {
        cRespGroup := 0.U
        cRespRow   := cRespRow + 1.U
      }.otherwise(cRespGroup := cRespGroup + 1.U)
    }.otherwise {
      when(aRespPart === 2.U) {
        aScale(aRespRow) := (readA.io.resp.bits.data >> scaleOffset)(7, 0)
        aRespPart        := 0.U
        aRespRow         := aRespRow + 1.U
      }.otherwise {
        aCodes(aRespRow)(aRespPart(0)) := readA.io.resp.bits.data
        aRespPart                      := aRespPart + 1.U
      }
    }
  }
  when(readB.io.req.fire) {
    when(bIssuePart === 2.U) {
      bIssuePart := 0.U
      bIssueRow  := bIssueRow + 1.U
      bIssueAddr := bIssueAddr + stride
    }.otherwise(bIssuePart := bIssuePart + 1.U)
  }
  when(readB.io.resp.fire) {
    when(bRespPart === 2.U) {
      bScale(bRespRow) := (readB.io.resp.bits.data >> scaleOffset)(7, 0)
      bRespPart        := 0.U
      bRespRow         := bRespRow + 1.U
    }.otherwise {
      bCodes(bRespRow)(bRespPart(0)) := readB.io.resp.bits.data
      bRespPart                      := bRespPart + 1.U
    }
  }
  when(write.io.req.fire) {
    when(cIssueGroup === cGroups - 1.U) {
      cIssueGroup := 0.U
      cIssueRow   := cIssueRow + 1.U
      cIssueAddr  := cIssueAddr + (n >> 2)
    }.otherwise(cIssueGroup := cIssueGroup + 1.U)
  }
  when(write.io.resp.fire) {
    assert(write.io.resp.bits.ok, "MXINT8MM bank write failed")
    when(cRespGroup === cGroups - 1.U) {
      cRespGroup := 0.U
      cRespRow   := cRespRow + 1.U
    }.otherwise(cRespGroup := cRespGroup + 1.U)
  }

  switch(state) {
    is(idle) {
      when(blink.cmdReq.fire) {
        val cmd    = blink.cmdReq.bits
        val count  = cmd.iter
        val mCount = cmd.rs2(9, 0)
        val nCount = cmd.rs2(19, 10)
        val aStart = cmd.rs2(33, 20)
        val bStart = cmd.rs2(47, 34)
        val cStart = cmd.rs2(61, 48)
        val packed = (count >> 4) +& (count >> 9) +& (count(8, 0) =/= 0.U)
        assert(cmd.rs2(63) === 0.U, "MXINT8MM reserved fields must be zero")
        assert(mCount =/= 0.U && nCount =/= 0.U && nCount(1, 0) === 0.U, "MXINT8MM M > 0 and N a positive multiple of 4")
        assert(count =/= 0.U && count(4, 0) === 0.U, "MXINT8MM K must be a positive multiple of 32")
        assert(
          cmd.op1_bank =/= cmd.op2_bank && cmd.op1_bank =/= cmd.wr_bank && cmd.op2_bank =/= cmd.wr_bank,
          "MXINT8MM banks must differ"
        )
        assert(cmd.op1_col === 1.U && cmd.op2_col === 1.U && cmd.wr_col === 1.U, "MXINT8MM banks must have one group")
        assert(aStart +& mCount * packed <= mp.bankEntries.U, "MXINT8MM A window exceeds capacity")
        assert(bStart +& nCount * packed <= mp.bankEntries.U, "MXINT8MM B window exceeds capacity")
        assert(cStart +& mCount * (nCount >> 2) <= mp.bankEntries.U, "MXINT8MM C window exceeds capacity")
        robId   := cmd.rob_id
        aBank   := cmd.op1_bank
        bBank   := cmd.op2_bank
        cBank   := cmd.wr_bank
        oldBank := cmd.wr_read_bank
        aGroup  := cmd.op1_group
        bGroup  := cmd.op2_group
        cGroup  := cmd.wr_group
        m       := mCount
        n       := nCount
        k       := count
        stride  := packed
        bBase   := bStart
        fromOld := cmd.rs2(62)
        m0      := 0.U
        n0      := 0.U
        aTile   := aStart
        bTile   := bStart
        cTile   := cStart
        state   := tile
      }
    }
    is(tile) {
      aIssueRow   := 0.U
      aIssuePart  := 0.U
      aIssueAddr  := aTile
      aRespRow    := 0.U
      aRespPart   := 0.U
      bIssueRow   := 0.U
      bIssuePart  := 0.U
      bIssueAddr  := bTile
      bRespRow    := 0.U
      bRespPart   := 0.U
      cIssueRow   := 0.U
      cIssueGroup := 0.U
      cIssueAddr  := cTile + (n0 >> 2)
      cRespRow    := 0.U
      cRespGroup  := 0.U
      block       := 0.U
      state       := Mux(fromOld, oldC, load)
    }
    is(oldC)(when(cRespRow === activeRows)(state := load))
    is(load) {
      when(aRespRow === activeRows && bRespRow === activeCols) {
        step  := 0.U
        state := mac
      }
    }
    is(mac) {
      step                               := step + 1.U
      when(step === (steps - 1).U)(state := convert)
    }
    is(convert)(state := accumulate)
    is(accumulate) {
      aIssueRow   := 0.U
      aIssuePart  := 0.U
      aIssueAddr  := aTile
      aRespRow    := 0.U
      aRespPart   := 0.U
      bIssueRow   := 0.U
      bIssuePart  := 0.U
      bIssueAddr  := bTile
      bRespRow    := 0.U
      bRespPart   := 0.U
      cIssueRow   := 0.U
      cIssueGroup := 0.U
      cIssueAddr  := cTile + (n0 >> 2)
      cRespRow    := 0.U
      cRespGroup  := 0.U
      when(block === (k >> 5) - 1.U)(state := store).otherwise {
        block := block + 1.U
        state := load
      }
    }
    is(store) {
      when(cRespRow === activeRows) {
        state := tile
        when(n0 +& cols.U >= n) {
          n0    := 0.U
          bTile := bBase
          when(m0 +& rows.U >= m)(state := complete).otherwise {
            m0    := m0 + rows.U
            aTile := aTile + stride * rows.U
            cTile := cTile + (n >> 2) * rows.U
          }
        }.otherwise {
          n0    := n0 + cols.U
          bTile := bTile + stride * cols.U
        }
      }
    }
    is(complete)(when(blink.cmdResp.fire)(state := idle))
  }
}

object Mxint8matmulBall extends BallFactory {

  def instantiate(
    fp: FrontendParam,
    mp: MemDomainParam,
    bp: BallDomainParam,
    id: Int
  ): BallPorts =
    BallPorts(Instantiate(new Mxint8matmulBall(fp, mp, bp, id)).blink, None)

}
