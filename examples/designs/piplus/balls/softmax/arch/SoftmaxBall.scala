package examples.designs.piplus.balls.softmax

import framework.generic.contract.blink.BlinkParam
import framework.generic.contract.configs.BallRouteParam
import framework.ball.sub_rob.ExpansionParam

import framework.ball.bbus.{BallFactory, BallPorts}
import buckyball.balls.softmax.{Params => ParametersProto}
import framework.ball.blink.{BlinkIO, HasBlink}

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import hardfloat._
import framework.generic.queue.Queue

/**
 * SOFTMAX: the softmax of FP32 rows scaled by 2^scale_log2 (workloads/isa/softmax.h; the BEMU model
 * emu/src/softmax.rs is the reference): each value is scaled as it is read.
 * Per row, three passes stream its bank rows, one bank row (four lanes) a cycle: pass 0 the maximum, pass 1 the
 * exponentials into four lane sums, pass 2 the exponentials again, times 1 / s, written. The reciprocal starts when
 * pass 1 ends; pass 2 writes once it is there.
 */
@instantiable
class SoftmaxBall(
  link:  BlinkParam,
  route: BallRouteParam)
    extends Module
    with HasBlink {
  val ball = route

  ball.parameters.unpack(classOf[ParametersProto])

  require(link.bankWidth == 128 && link.bankMaskLen == 16, "SoftmaxBall requires 128-bit bank rows and byte masks")
  require(ball.inBW == 1 && ball.outBW == 1, "SoftmaxBall requires one read and one write")
  require(link.dataBits == 64, "SoftmaxBall decodes a 64-bit rs2")

  @public
  val blink = IO(new BlinkIO(link, route))

  val rne   = consts.round_near_even
  val after = consts.tininess_afterRounding

  val busy        = RegInit(false.B)
  val done        = RegInit(false.B)
  val pass        = Reg(UInt(2.W))
  val robId       = Reg(UInt(link.robIdBits.W))
  val source      = Reg(UInt(link.bankIdBits.W))
  val destination = Reg(UInt(link.bankIdBits.W))
  val sourceGroup = Reg(UInt(link.groupIdBits.W))
  val targetGroup = Reg(UInt(link.groupIdBits.W))
  val rows        = Reg(UInt(link.iterBits.W))
  val w           = Reg(UInt(16.W))
  // 2^scale_log2, recoded: each row is scaled by it as it is read.
  val scale       = Reg(UInt(33.W))
  val lines       = Reg(UInt(16.W))
  val inRow       = Reg(UInt(32.W))
  val outRow      = Reg(UInt(32.W))
  val row         = Reg(UInt(link.iterBits.W))
  val issued      = Reg(UInt(16.W))
  val received    = Reg(UInt(16.W))
  val produced    = Reg(UInt(16.W))
  val written     = Reg(UInt(16.W))
  // Lane maxima as ordered keys: IEEE bits with the sign bit set, negatives complemented.
  val laneMax     = Reg(Vec(4, UInt(32.W)))
  val rowMax      = Reg(UInt(33.W))
  val laneSum     = Reg(Vec(4, UInt(33.W)))
  val inverse     = Reg(UInt(33.W))
  val inverseOk   = Reg(Bool())
  val divStarted  = Reg(Bool())

  val active = !reset.asBool
  blink.cmdReq.ready            := !busy && !done && active
  blink.cmdResp.valid           := done && active
  blink.cmdResp.bits.rob_id     := robId
  blink.status.idle             := !busy && !done
  blink.status.running          := busy
  when(blink.cmdResp.fire)(done := false.B)

  val write   = blink.bankWrite(0)
  // Pass 2 results wait for their write here: each lane its mask bit above its recoded e. The pipelines hold only
  // while it is full, a registered condition, so no ready of the bank ports depends on another.
  val results = Instantiate(new Queue(Vec(4, UInt(34.W)), 2))
  val advance = pass =/= 2.U || results.io.enq.ready

  val read = blink.bankRead(0)
  read.rob_id                      := robId
  read.bank_id                     := source
  read.group_id                    := sourceGroup
  read.io.req.valid                := busy && issued < lines && active
  read.io.req.bits.addr            := inRow + issued
  read.io.resp.ready               := busy && advance && active
  when(read.io.req.fire)(issued    := issued + 1.U)
  when(read.io.resp.fire)(received := received + 1.U)

  // Lane l of bank row j holds element 4j + l; lanes from w on are outside the row.
  val words = VecInit((0 until 4).map(l => read.io.resp.bits.data(32 * l + 31, 32 * l)))
  val masks = VecInit((0 until 4).map(l => Cat(received, l.U(2.W)) < w))

  // x = in * 2^scale_log2, an FP32 multiplication.
  val scalers = Seq.fill(4)(Module(new MulRecFN(8, 24)))
  for (l <- 0 until 4) {
    scalers(l).io.a              := recFNFromFN(8, 24, words(l))
    scalers(l).io.b              := scale
    scalers(l).io.roundingMode   := rne
    scalers(l).io.detectTininess := after
  }
  val values = VecInit(scalers.map(scaler => fNFromRecFN(8, 24, scaler.io.out)))

  val keys = VecInit(values.map(x => Mux(x(31), ~x, x | "h80000000".U(32.W))))
  when(pass === 0.U && read.io.resp.fire) {
    for (l <- 0 until 4) {
      assert(
        !masks(l) || values(l)(30, 23) =/= 255.U || values(l) === "hff800000".U,
        "SOFTMAX value neither finite nor -inf"
      )
      when(masks(l) && keys(l) > laneMax(l))(laneMax(l) := keys(l))
    }
  }

  val lanes = Seq.fill(4)(Instantiate(new Exp))
  for (l <- 0 until 4) {
    lanes(l).io.advance := advance
    lanes(l).io.inValid := read.io.resp.fire && pass =/= 0.U
    lanes(l).io.inMask  := masks(l)
    lanes(l).io.x       := scalers(l).io.out
    lanes(l).io.m       := rowMax
  }
  val laneValid = lanes.head.io.outValid

  // Pass 1: s_l += e_l in increasing element order.
  val sums = Seq.fill(4)(Module(new AddRecFN(8, 24)))
  for (l <- 0 until 4) {
    sums(l).io.subOp          := false.B
    sums(l).io.a              := laneSum(l)
    sums(l).io.b              := lanes(l).io.e
    sums(l).io.roundingMode   := rne
    sums(l).io.detectTininess := after
  }
  when(pass === 1.U && laneValid) {
    for (l <- 0 until 4)(when(lanes(l).io.outMask)(laneSum(l) := sums(l).io.out))
    produced := produced + 1.U
  }

  // 1 / ((s_0 + s_1) + (s_2 + s_3)) while pass 2 fills its pipelines.
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
  val divider = Module(new DivSqrtRecFN_small(8, 24, 0))
  divider.io.inValid                                        := pass === 2.U && !divStarted
  divider.io.sqrtOp                                         := false.B
  divider.io.a                                              := recFNFromFN(8, 24, "h3f800000".U(32.W))
  divider.io.b                                              := pairs(2).io.out
  divider.io.roundingMode                                   := rne
  divider.io.detectTininess                                 := after
  when(divider.io.inValid && divider.io.inReady)(divStarted := true.B)
  when(divider.io.outValid_div) {
    inverse   := divider.io.out
    inverseOk := true.B
  }

  // Pass 2: y = e * (1 / s); lanes outside the row write zero.
  results.io.enq.valid := pass === 2.U && laneValid
  results.io.enq.bits  := VecInit(lanes.map(lane => Cat(lane.io.outMask, lane.io.e)))
  results.io.deq.ready := inverseOk && write.io.req.ready
  val result   = results.io.deq.bits
  val products = Seq.fill(4)(Module(new MulRecFN(8, 24)))
  for (l <- 0 until 4) {
    products(l).io.a              := result(l)(32, 0)
    products(l).io.b              := inverse
    products(l).io.roundingMode   := rne
    products(l).io.detectTininess := after
  }
  write.rob_id := robId
  write.bank_id                    := destination
  write.group_id                   := targetGroup
  write.io.req.valid               := results.io.deq.valid && inverseOk && active
  write.io.req.bits.addr           := outRow + produced
  write.io.req.bits.data           := Cat((0 until 4).reverse.map(l =>
    Mux(result(l)(33), fNFromRecFN(8, 24, products(l).io.out), 0.U(32.W))
  ))
  write.io.req.bits.mask           := VecInit(Seq.fill(16)(true.B))
  write.io.resp.ready              := active
  when(write.io.req.fire)(produced := produced + 1.U)
  when(write.io.resp.fire) {
    assert(write.io.resp.bits.ok, "SOFTMAX bank write failed")
    written := written + 1.U
  }

  // The end of a pass restarts the counters on the same bank rows.
  val passDone = busy && MuxLookup(pass, false.B)(
    Seq(0.U -> (received === lines), 1.U -> (produced === lines), 2.U -> (written === lines))
  )

  val best = laneMax.reduce((a, b) => Mux(a > b, a, b))
  when(passDone) {
    issued   := 0.U
    received := 0.U
    produced := 0.U
    written  := 0.U
    pass     := Mux(pass === 2.U, 0.U, pass + 1.U)
    when(pass === 0.U) {
      assert(best =/= "h007fffff".U, "SOFTMAX a row without a finite value")
      rowMax  := recFNFromFN(8, 24, Mux(best(31), best & "h7fffffff".U(32.W), ~best))
      laneSum := VecInit(Seq.fill(4)(0.U(33.W)))
    }
    when(pass === 1.U) {
      divStarted := false.B
      inverseOk  := false.B
    }
    when(pass === 2.U) {
      inRow   := inRow + lines
      outRow  := outRow + lines
      row     := row + 1.U
      laneMax := VecInit(Seq.fill(4)(0.U(32.W)))
      when(row === rows - 1.U) {
        busy := false.B
        done := true.B
      }
    }
  }

  when(blink.cmdReq.fire) {
    val cmd      = blink.cmdReq.bits
    val count    = cmd.rs2(15, 0)
    val inBase   = cmd.rs2(29, 16)
    val outBase  = cmd.rs2(43, 30)
    val perRow   = (count +& 3.U) >> 2
    val exponent = cmd.rs2(51, 44).asSInt
    assert(cmd.rs2(63, 52) === 0.U, "SOFTMAX reserved fields must be zero")
    assert(exponent >= -126.S, "SOFTMAX scale exponent below -126")
    assert(cmd.iter =/= 0.U && count >= 16.U && count <= 2048.U, "SOFTMAX rows > 0 and 16 <= W <= 2048")
    assert(cmd.op1_bank =/= cmd.wr_bank, "SOFTMAX input and output banks must differ")
    assert(cmd.op1_col === 1.U && cmd.wr_col === 1.U, "SOFTMAX banks must have one group")
    assert(inBase +& cmd.iter * perRow <= link.bankEntries.U, "SOFTMAX input exceeds capacity")
    assert(outBase +& cmd.iter * perRow <= link.bankEntries.U, "SOFTMAX output exceeds capacity")
    busy        := true.B
    pass        := 0.U
    robId       := cmd.rob_id
    source      := cmd.op1_bank
    destination := cmd.wr_bank
    sourceGroup := cmd.op1_group
    targetGroup := cmd.wr_group
    rows        := cmd.iter
    w           := count
    scale       := recFNFromFN(8, 24, Cat(0.U(1.W), (exponent + 127.S).asUInt(7, 0), 0.U(23.W)))
    lines       := perRow
    inRow       := inBase
    outRow      := outBase
    row         := 0.U
    issued      := 0.U
    received    := 0.U
    produced    := 0.U
    written     := 0.U
    laneMax     := VecInit(Seq.fill(4)(0.U(32.W)))
  }
}

object SoftmaxBall extends BallFactory {

  def instantiate(
    link:      BlinkParam,
    route:     BallRouteParam,
    expansion: Option[ExpansionParam]
  ): BallPorts =
    BallPorts(Instantiate(new SoftmaxBall(link, route)).blink, None)

}
