package examples.balls.nli

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public}
import chisel3.util._
import framework.balldomain.blink.{BallStatus, BlinkIO, HasBallStatus, HasBlink, SubRobRow}
import framework.balldomain.blink.mmio.{MmioRead, MmioWrite}
import framework.top.GlobalConfig

/**
 * NliBall - Non-uniform Linear Interpolation (NLI) accelerator.
 *
 * NLI approximates a non-linear function f(x) with a small number of
 * piecewise-linear segments whose cutpoints are placed NON-uniformly (offline,
 * where the function is most curved). For each INT8 input lane the ball:
 *
 *   1. selects the segment i with `x >= cutpoint[i]` over the 15 interior
 *      cutpoints (16 segments tiling the signed INT8 domain),
 *   2. computes `out = clamp((slope[i] * x) >> 7 + intercept[i], -128, 127)`.
 *
 * `slope` is Q7 fixed point (slope / 128); `>> 7` is an arithmetic shift so the
 * emulator, the RTL and the C reference model agree bit-for-bit.
 *
 * The segment table is generic: any function (SiLU, GELU, exp, ...) is
 * described purely by its coefficient bank, loaded with MVIN before issue.
 * Coefficient bank layout (one bank, col = 1, three rows):
 *   row 0 : 15 cutpoints (INT8) in bytes 0..14, byte 15 unused
 *   row 1 : 16 slopes    (INT8, Q7) in bytes 0..15
 *   row 2 : 16 intercepts (INT8)    in bytes 0..15
 *
 * Data layout mirrors LutBall: 16 INT8 lanes per 128-bit bank row.
 */
@instantiable
class NliBall(val b: GlobalConfig) extends Module with HasBlink with HasBallStatus {

  private val mapping = b.ballDomain.ballIdMappings
    .find(_.ballName == "NliBall")
    .getOrElse(throw new IllegalArgumentException("NliBall not found in config"))

  private val funct = b.ballDomain.ballISA
    .find(_.mnemonic == "NLI")
    .map(_.funct7)
    .getOrElse(throw new IllegalArgumentException("NLI not found in ballISA"))

  require(b.memDomain.bankWidth == 128, "NliBall requires 128-bit bank rows")
  require(b.memDomain.bankEntries >= 16, "NliBall requires at least 16 rows per bank")
  require(mapping.inBW == 2, "NliBall requires inBW=2")
  require(mapping.outBW == 1, "NliBall requires outBW=1")
  require((funct >> 4) == 4, "NLI must encode two reads and one write")

  @public val io = IO(new BlinkIO(b, mapping.inBW, mapping.outBW))
  def blink:  BlinkIO    = io
  def status: BallStatus = io.status
  dontTouch(io)

  private val idle :: coeffReq :: coeffResp :: inputReq :: inputResp :: compute :: writeReq :: writeResp :: complete :: Nil =
    Enum(9)

  private val state      = RegInit(idle)
  private val robId      = RegInit(0.U(log2Up(b.frontend.rob_entries).W))
  private val isSub      = RegInit(false.B)
  private val subRobId   = RegInit(0.U(log2Up(b.frontend.sub_rob_depth * 4).W))
  private val inputBank  = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  private val coeffBank  = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  private val outputBank = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  private val iter       = RegInit(0.U(b.frontend.iter_len.W))
  private val inputRow   = RegInit(0.U(log2Ceil(b.memDomain.bankEntries).W))
  private val coeffRow   = RegInit(0.U(2.W))
  private val inputData  = Reg(UInt(128.W))

  // NLI segment table, latched from the coefficient bank (3 rows).
  private val cutpoints  = Reg(Vec(15, SInt(8.W)))
  private val slopes     = Reg(Vec(16, SInt(8.W)))
  private val intercepts = Reg(Vec(16, SInt(8.W)))
  private val outputWord = Reg(Vec(16, UInt(8.W)))

  io.cmdReq.ready            := state === idle
  io.cmdResp.valid           := state === complete
  io.cmdResp.bits.rob_id     := robId
  io.cmdResp.bits.is_sub     := isSub
  io.cmdResp.bits.sub_rob_id := subRobId
  io.status.idle             := state === idle
  io.status.running          := state =/= idle && state =/= complete

  for (port <- 0 until 2) {
    io.bankRead(port).rob_id           := robId
    io.bankRead(port).ball_id          := 0.U
    io.bankRead(port).group_id         := 0.U
    io.bankRead(port).io.req.valid     := false.B
    io.bankRead(port).io.req.bits.addr := 0.U
    io.bankRead(port).io.resp.ready    := false.B
  }
  io.bankRead(0).bank_id := inputBank
  io.bankRead(1).bank_id := coeffBank

  io.bankWrite(0).rob_id           := robId
  io.bankWrite(0).ball_id          := 0.U
  io.bankWrite(0).bank_id          := outputBank
  io.bankWrite(0).group_id         := 0.U
  io.bankWrite(0).io.req.valid     := false.B
  io.bankWrite(0).io.req.bits.addr := inputRow
  io.bankWrite(0).io.req.bits.data := Cat(outputWord.reverse)
  io.bankWrite(0).io.req.bits.mask := VecInit(Seq.fill(b.memDomain.bankMaskLen)(true.B))
  io.bankWrite(0).io.resp.ready    := false.B

  io.subRobReq.valid := false.B
  io.subRobReq.bits  := SubRobRow.tieOff(b)
  MmioRead.tieOff(io.mmioRead)
  MmioWrite.tieOff(io.mmioWrite)

  /** One NLI lane: select segment by comparator tree, then one Q7 MAC. */
  private def mac(lane: Int): UInt = {
    val x         = inputData(8 * lane + 7, 8 * lane).asSInt
    val ge        = (0 until 15).map(j => x >= cutpoints(j))
    val seg       = PopCount(VecInit(ge))
    val slope     = slopes(seg)
    val intercept = intercepts(seg)
    val prod      = slope * x           // SInt(16.W)
    val scaled    = prod >> 7           // arithmetic shift (Q7 -> INT8)
    val acc       = scaled +& intercept // widened sum
    val clamped   = Mux(acc > 127.S, 127.S, Mux(acc < -128.S, -128.S, acc))
    clamped(7, 0) // two's-complement low byte
  }

  switch(state) {
    is(idle) {
      when(io.cmdReq.fire) {
        val cmd = io.cmdReq.bits.cmd
        assert(cmd.funct7 === funct.U, "NliBall funct7 must be NLI")
        assert(cmd.rs2 === 0.U, "NliBall reserves rs2")
        assert(cmd.iter > 0.U && cmd.iter <= b.memDomain.bankEntries.U, "NliBall iter must fit in one bank")
        assert(
          cmd.op1_col === 1.U && cmd.op2_col === 1.U && cmd.wr_col === 1.U,
          "NliBall requires col=1 input/table/output"
        )
        assert(
          cmd.op1_bank =/= cmd.op2_bank && cmd.op1_bank =/= cmd.wr_bank &&
            cmd.op2_bank =/= cmd.wr_bank,
          "NliBall banks must be distinct"
        )
        robId      := io.cmdReq.bits.rob_id
        isSub      := io.cmdReq.bits.is_sub
        subRobId   := io.cmdReq.bits.sub_rob_id
        inputBank  := cmd.op1_bank
        coeffBank  := cmd.op2_bank
        outputBank := cmd.wr_bank
        iter       := cmd.iter
        inputRow   := 0.U
        coeffRow   := 0.U
        state      := coeffReq
      }
    }
    is(coeffReq) {
      io.bankRead(1).io.req.valid            := true.B
      io.bankRead(1).io.req.bits.addr        := coeffRow
      when(io.bankRead(1).io.req.fire)(state := coeffResp)
    }
    is(coeffResp) {
      io.bankRead(1).io.resp.ready := true.B
      when(io.bankRead(1).io.resp.fire) {
        val data = io.bankRead(1).io.resp.bits.data
        when(coeffRow === 0.U) {
          for (j <- 0 until 15) { cutpoints(j) := data(8 * j + 7, 8 * j).asSInt }
        }.elsewhen(coeffRow === 1.U) {
          for (j <- 0 until 16) { slopes(j) := data(8 * j + 7, 8 * j).asSInt }
        }.otherwise {
          for (j <- 0 until 16) { intercepts(j) := data(8 * j + 7, 8 * j).asSInt }
        }
        when(coeffRow === 2.U)(state := inputReq)
          .otherwise { coeffRow := coeffRow + 1.U; state := coeffReq }
      }
    }
    is(inputReq) {
      io.bankRead(0).io.req.valid            := true.B
      io.bankRead(0).io.req.bits.addr        := inputRow
      when(io.bankRead(0).io.req.fire)(state := inputResp)
    }
    is(inputResp) {
      io.bankRead(0).io.resp.ready := true.B
      when(io.bankRead(0).io.resp.fire) {
        inputData := io.bankRead(0).io.resp.bits.data
        state     := compute
      }
    }
    is(compute) {
      for (lane <- 0 until 16) { outputWord(lane) := mac(lane) }
      state := writeReq
    }
    is(writeReq) {
      io.bankWrite(0).io.req.valid            := true.B
      when(io.bankWrite(0).io.req.fire)(state := writeResp)
    }
    is(writeResp) {
      io.bankWrite(0).io.resp.ready := true.B
      when(io.bankWrite(0).io.resp.fire) {
        when(inputRow === iter - 1.U)(state := complete)
          .otherwise { inputRow := inputRow + 1.U; state := inputReq }
      }
    }
    is(complete) {
      when(io.cmdResp.fire)(state := idle)
    }
  }
}
