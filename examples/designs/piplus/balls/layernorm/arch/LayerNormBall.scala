package examples.designs.piplus.balls.layernorm

import framework.generic.contract.blink.BlinkParam
import framework.generic.contract.configs.BallRouteParam
import framework.ball.sub_rob.ExpansionParam

import framework.ball.bbus.{BallFactory, BallPorts}
import buckyball.balls.layernorm.{Params => ParametersProto}
import framework.ball.blink.{BlinkIO, HasBlink}

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import hardfloat._
import framework.generic.queue.Queue

/**
 * LAYERNORM: the layer normalization of FP32 rows (workloads/isa/layernorm.h; the BEMU model emu/src/layernorm.rs is
 * the reference). Per row, three passes stream its bank rows on read port 0, four lanes a bank row: pass 0 the lane
 * sums of x (port 1 reads eps once), pass 1 the lane sums of the squared deviations, pass 2 the output, gamma and
 * beta on port 1, two bank rows per input row. 1 / w is divided at the start of an instruction, sqrt and the
 * reciprocal while pass 2 starts reading. With rms (RMSNorm) pass 0 reads no x (the mean is 0) and pass 2 reads gamma
 * only, the output (x * r) * gamma.
 */
@instantiable
class LayerNormBall(
  link:  BlinkParam,
  route: BallRouteParam)
    extends Module
    with HasBlink {
  val ball = route

  ball.parameters.unpack(classOf[ParametersProto])

  require(link.bankWidth == 128 && link.bankMaskLen == 16, "LayerNormBall requires 128-bit bank rows and byte masks")
  require(ball.inBW == 2 && ball.outBW == 1, "LayerNormBall requires two reads and one write")
  require(link.dataBits == 64, "LayerNormBall decodes a 64-bit rs2")

  @public
  val blink = IO(new BlinkIO(link, route))

  val rne   = consts.round_near_even
  val after = consts.tininess_afterRounding

  val busy        = RegInit(false.B)
  val done        = RegInit(false.B)
  val pass        = Reg(UInt(2.W))
  val rms         = Reg(Bool())
  val robId       = Reg(UInt(link.robIdBits.W))
  val source      = Reg(UInt(link.bankIdBits.W))
  val params      = Reg(UInt(link.bankIdBits.W))
  val destination = Reg(UInt(link.bankIdBits.W))
  val sourceGroup = Reg(UInt(link.groupIdBits.W))
  val paramGroup  = Reg(UInt(link.groupIdBits.W))
  val targetGroup = Reg(UInt(link.groupIdBits.W))
  val rows        = Reg(UInt(link.iterBits.W))
  val w           = Reg(UInt(16.W))
  val lines       = Reg(UInt(16.W))
  val inRow       = Reg(UInt(32.W))
  val outRow      = Reg(UInt(32.W))
  val paramBase   = Reg(UInt(32.W))
  val row         = Reg(UInt(link.iterBits.W))
  val issued      = Reg(UInt(16.W))
  val received    = Reg(UInt(16.W))
  val issued1     = Reg(UInt(17.W))
  val produced    = Reg(UInt(16.W))
  val written     = Reg(UInt(16.W))
  val laneSum     = Reg(Vec(4, UInt(33.W)))
  val mean        = Reg(UInt(33.W))
  val inverse     = Reg(UInt(33.W))
  val inverseOk   = Reg(Bool())
  val invStarted  = Reg(Bool())
  val eps         = Reg(UInt(33.W))
  val epsOk       = Reg(Bool())
  val epsIssued   = Reg(Bool())
  val root        = Reg(UInt(33.W))
  val rootOk      = Reg(Bool())
  val sqrtStarted = Reg(Bool())
  val divStarted  = Reg(Bool())
  val r           = Reg(UInt(33.W))
  val rOk         = RegInit(false.B)
  // Pass 2 operands of one bank row: x, gamma, beta.
  val xRow        = Reg(UInt(128.W))
  val xOk         = RegInit(false.B)
  val xMask       = Reg(Vec(4, Bool()))
  val gammaRow    = Reg(UInt(128.W))
  val gammaOk     = RegInit(false.B)
  val betaRow     = Reg(UInt(128.W))
  val betaOk      = RegInit(false.B)

  val active = !reset.asBool
  blink.cmdReq.ready            := !busy && !done && active
  blink.cmdResp.valid           := done && active
  blink.cmdResp.bits.rob_id     := robId
  blink.status.idle             := !busy && !done
  blink.status.running          := busy
  when(blink.cmdResp.fire)(done := false.B)

  // Port 0: x, every pass; one row at a time in pass 2.
  val read = blink.bankRead(0)
  read.rob_id                      := robId
  read.bank_id                     := source
  read.group_id                    := sourceGroup
  read.io.req.valid                := busy && issued < lines && !(pass === 0.U && rms) &&
    (pass =/= 2.U || issued < produced +& 2.U) && active
  read.io.req.bits.addr            := inRow + issued
  read.io.resp.ready               := busy && (pass =/= 2.U || !xOk) && active
  when(read.io.req.fire)(issued    := issued + 1.U)
  when(read.io.resp.fire)(received := received + 1.U)

  // Lane l of bank row j holds element 4j + l; lanes from w on are outside the row.
  val words = VecInit((0 until 4).map(l => recFNFromFN(8, 24, read.io.resp.bits.data(32 * l + 31, 32 * l))))
  val masks = VecInit((0 until 4).map(l => Cat(received, l.U(2.W)) < w))

  // Port 1: eps in pass 0, then gamma and beta alternately in pass 2.
  val readParams = blink.bankRead(1)
  val gammaTurn  = !issued1(0)
  readParams.rob_id           := robId
  readParams.bank_id          := params
  readParams.group_id         := paramGroup
  readParams.io.req.valid     := busy && active &&
    Mux(pass === 0.U, !epsIssued, pass === 2.U && issued1 < (lines << 1) && issued1 < (produced << 1) +& 4.U)
  readParams.io.req.bits.addr := Mux(
    pass === 0.U,
    paramBase + Mux(rms, lines, lines << 1),
    paramBase + (issued1 >> 1) + Mux(gammaTurn, 0.U, lines)
  )
  readParams.io.resp.ready    := busy && active && Mux(pass === 0.U, true.B, !gammaOk || (!rms && !betaOk))
  when(readParams.io.req.fire) {
    when(pass === 0.U)(epsIssued := true.B).otherwise(issued1 := issued1 + Mux(rms, 2.U, 1.U))
  }
  when(readParams.io.resp.fire) {
    when(pass === 0.U) {
      eps   := recFNFromFN(8, 24, readParams.io.resp.bits.data(31, 0))
      epsOk := true.B
    }.elsewhen(!gammaOk) {
      gammaRow := readParams.io.resp.bits.data
      gammaOk  := true.B
    }.otherwise {
      betaRow := readParams.io.resp.bits.data
      betaOk  := true.B
    }
  }

  // Pass 0: s_l += x; pass 1: q_l += (x - mean)^2, in increasing element order.
  val deviations = Seq.fill(4)(Module(new AddRecFN(8, 24)))
  val squares    = Seq.fill(4)(Module(new MulRecFN(8, 24)))
  val sums       = Seq.fill(4)(Module(new AddRecFN(8, 24)))
  for (l <- 0 until 4) {
    deviations(l).io.subOp          := true.B
    deviations(l).io.a              := words(l)
    deviations(l).io.b              := mean
    deviations(l).io.roundingMode   := rne
    deviations(l).io.detectTininess := after
    squares(l).io.a                 := deviations(l).io.out
    squares(l).io.b                 := deviations(l).io.out
    squares(l).io.roundingMode      := rne
    squares(l).io.detectTininess    := after
    sums(l).io.subOp                := false.B
    sums(l).io.a                    := laneSum(l)
    sums(l).io.b                    := Mux(pass === 0.U, words(l), squares(l).io.out)
    sums(l).io.roundingMode         := rne
    sums(l).io.detectTininess       := after
  }
  when(pass =/= 2.U && read.io.resp.fire) {
    for (l <- 0 until 4)(when(masks(l))(laneSum(l) := sums(l).io.out))
  }

  // (s_0 + s_1) + (s_2 + s_3), times 1 / w.
  val pairs = Seq.fill(3)(Module(new AddRecFN(8, 24)))
  pairs.foreach { adder =>
    adder.io.subOp          := false.B
    adder.io.roundingMode   := rne
    adder.io.detectTininess := after
  }
  pairs(0).io.a := laneSum(0)
  pairs(0).io.b := laneSum(1)
  pairs(1).io.a := laneSum(2)
  pairs(1).io.b := laneSum(3)
  pairs(2).io.a := pairs(0).io.out
  pairs(2).io.b := pairs(1).io.out
  val scaled = Module(new MulRecFN(8, 24))
  scaled.io.a              := pairs(2).io.out
  scaled.io.b              := inverse
  scaled.io.roundingMode   := rne
  scaled.io.detectTininess := after
  // var + eps
  val shifted = Module(new AddRecFN(8, 24))
  shifted.io.subOp          := false.B
  shifted.io.a              := scaled.io.out
  shifted.io.b              := eps
  shifted.io.roundingMode   := rne
  shifted.io.detectTininess := after

  // 1 / w at the start of an instruction.
  val wDivider = Module(new DivSqrtRecFN_small(8, 24, 0))
  val wFloat   = Module(new INToRecFN(16, 8, 24))
  wFloat.io.signedIn                                          := false.B
  wFloat.io.in                                                := w
  wFloat.io.roundingMode                                      := rne
  wFloat.io.detectTininess                                    := after
  wDivider.io.inValid                                         := busy && !invStarted
  wDivider.io.sqrtOp                                          := false.B
  wDivider.io.a                                               := recFNFromFN(8, 24, "h3f800000".U(32.W))
  wDivider.io.b                                               := wFloat.io.out
  wDivider.io.roundingMode                                    := rne
  wDivider.io.detectTininess                                  := after
  when(wDivider.io.inValid && wDivider.io.inReady)(invStarted := true.B)
  when(wDivider.io.outValid_div) {
    inverse   := wDivider.io.out
    inverseOk := true.B
  }

  // sqrt(var + eps), then its reciprocal, while pass 2 reads.
  val sqrtUnit = Module(new DivSqrtRecFN_small(8, 24, 0))
  sqrtUnit.io.inValid                                          := pass === 2.U && !sqrtStarted
  sqrtUnit.io.sqrtOp                                           := true.B
  sqrtUnit.io.a                                                := shifted.io.out
  sqrtUnit.io.b                                                := shifted.io.out
  sqrtUnit.io.roundingMode                                     := rne
  sqrtUnit.io.detectTininess                                   := after
  when(sqrtUnit.io.inValid && sqrtUnit.io.inReady)(sqrtStarted := true.B)
  when(sqrtUnit.io.outValid_sqrt) {
    root   := sqrtUnit.io.out
    rootOk := true.B
  }
  val rDivider = Module(new DivSqrtRecFN_small(8, 24, 0))
  rDivider.io.inValid                                          := rootOk && !divStarted
  rDivider.io.sqrtOp                                           := false.B
  rDivider.io.a                                                := recFNFromFN(8, 24, "h3f800000".U(32.W))
  rDivider.io.b                                                := root
  rDivider.io.roundingMode                                     := rne
  rDivider.io.detectTininess                                   := after
  when(rDivider.io.inValid && rDivider.io.inReady)(divStarted  := true.B)
  when(rDivider.io.outValid_div) {
    r   := rDivider.io.out
    rOk := true.B
  }

  // Pass 2: y = ((x - mean) * r) * gamma + beta; lanes outside the row write zero.
  when(pass === 2.U && read.io.resp.fire) {
    xRow  := read.io.resp.bits.data
    xOk   := true.B
    xMask := masks
  }
  val results  = Instantiate(new Queue(UInt(128.W), 2))
  val operands = xOk && gammaOk && (betaOk || rms) && rOk

  val outputs = (0 until 4).map { l =>
    val d = Module(new AddRecFN(8, 24))
    d.io.subOp          := true.B
    d.io.a              := recFNFromFN(8, 24, xRow(32 * l + 31, 32 * l))
    d.io.b              := mean
    d.io.roundingMode   := rne
    d.io.detectTininess := after
    val n = Module(new MulRecFN(8, 24))
    n.io.a              := d.io.out
    n.io.b              := r
    n.io.roundingMode   := rne
    n.io.detectTininess := after
    val g = Module(new MulRecFN(8, 24))
    g.io.a              := n.io.out
    g.io.b              := recFNFromFN(8, 24, gammaRow(32 * l + 31, 32 * l))
    g.io.roundingMode   := rne
    g.io.detectTininess := after
    val y = Module(new AddRecFN(8, 24))
    y.io.subOp          := false.B
    y.io.a              := g.io.out
    y.io.b              := recFNFromFN(8, 24, betaRow(32 * l + 31, 32 * l))
    y.io.roundingMode   := rne
    y.io.detectTininess := after
    Mux(xMask(l), fNFromRecFN(8, 24, Mux(rms, g.io.out, y.io.out)), 0.U(32.W))
  }

  results.io.enq.valid := operands
  results.io.enq.bits  := Cat(outputs.reverse)
  when(results.io.enq.fire) {
    xOk      := false.B
    gammaOk  := false.B
    betaOk   := false.B
    produced := produced + 1.U
  }

  val write = blink.bankWrite(0)
  write.rob_id           := robId
  write.bank_id          := destination
  write.group_id         := targetGroup
  write.io.req.valid     := results.io.deq.valid && active
  write.io.req.bits.data := results.io.deq.bits
  write.io.req.bits.mask := VecInit(Seq.fill(16)(true.B))
  write.io.resp.ready    := active
  results.io.deq.ready   := write.io.req.ready && active
  val writeIssued = Reg(UInt(16.W))
  when(write.io.req.fire)(writeIssued := writeIssued + 1.U)
  write.io.req.bits.addr              := outRow + writeIssued
  when(write.io.resp.fire) {
    assert(write.io.resp.bits.ok, "LAYERNORM bank write failed")
    written := written + 1.U
  }

  // The end of a pass restarts the counters on the same bank rows.
  val passDone = busy && MuxLookup(pass, false.B)(
    Seq(
      0.U -> ((received === lines || rms) && inverseOk && epsOk),
      1.U -> (received === lines),
      2.U -> (written === lines)
    )
  )

  when(passDone) {
    issued   := 0.U
    received := 0.U
    pass     := Mux(pass === 2.U, 0.U, pass + 1.U)
    when(pass === 0.U) {
      mean    := Mux(rms, 0.U, scaled.io.out)
      laneSum := VecInit(Seq.fill(4)(0.U(33.W)))
    }
    when(pass === 1.U) {
      sqrtStarted := false.B
      rootOk      := false.B
      divStarted  := false.B
      rOk         := false.B
      issued1     := 0.U
      produced    := 0.U
      written     := 0.U
      writeIssued := 0.U
    }
    when(pass === 2.U) {
      inRow   := inRow + lines
      outRow  := outRow + lines
      row     := row + 1.U
      laneSum := VecInit(Seq.fill(4)(0.U(33.W)))
      when(row === rows - 1.U) {
        busy := false.B
        done := true.B
      }
    }
  }

  when(blink.cmdReq.fire) {
    val cmd     = blink.cmdReq.bits
    val count   = cmd.rs2(15, 0)
    val inBase  = cmd.rs2(29, 16)
    val outBase = cmd.rs2(43, 30)
    val pBase   = cmd.rs2(57, 44)
    val perRow  = (count +& 3.U) >> 2
    val byRms   = cmd.rs2(58)
    assert(cmd.rs2(63, 59) === 0.U, "LAYERNORM reserved fields must be zero")
    assert(cmd.iter =/= 0.U && count >= 16.U && count <= 2048.U, "LAYERNORM rows > 0 and 16 <= W <= 2048")
    assert(
      cmd.op1_bank =/= cmd.wr_bank && cmd.op2_bank =/= cmd.wr_bank && cmd.op1_bank =/= cmd.op2_bank,
      "LAYERNORM banks must differ"
    )
    assert(cmd.op1_col === 1.U && cmd.op2_col === 1.U && cmd.wr_col === 1.U, "LAYERNORM banks must have one group")
    assert(inBase +& cmd.iter * perRow <= link.bankEntries.U, "LAYERNORM input exceeds capacity")
    assert(outBase +& cmd.iter * perRow <= link.bankEntries.U, "LAYERNORM output exceeds capacity")
    assert(pBase +& Mux(byRms, perRow, perRow << 1) +& 1.U <= link.bankEntries.U, "LAYERNORM parameters exceed capacity")
    busy        := true.B
    pass        := 0.U
    rms         := byRms
    robId       := cmd.rob_id
    source      := cmd.op1_bank
    params      := cmd.op2_bank
    destination := cmd.wr_bank
    sourceGroup := cmd.op1_group
    paramGroup  := cmd.op2_group
    targetGroup := cmd.wr_group
    rows        := cmd.iter
    w           := count
    lines       := perRow
    inRow       := inBase
    outRow      := outBase
    paramBase   := pBase
    row         := 0.U
    issued      := 0.U
    received    := 0.U
    laneSum     := VecInit(Seq.fill(4)(0.U(33.W)))
    inverseOk   := false.B
    invStarted  := false.B
    epsOk       := false.B
    epsIssued   := false.B
    xOk         := false.B
    gammaOk     := false.B
    betaOk      := false.B
  }
}

object LayerNormBall extends BallFactory {

  def instantiate(
    link:      BlinkParam,
    route:     BallRouteParam,
    expansion: Option[ExpansionParam]
  ): BallPorts =
    BallPorts(Instantiate(new LayerNormBall(link, route)).blink, None)

}
