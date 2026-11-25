package examples.designs.piplus.balls.mxint8quant

import framework.generic.contract.blink.BlinkParam
import framework.generic.contract.configs.BallRouteParam
import framework.ball.sub_rob.ExpansionParam

import framework.ball.bbus.{BallFactory, BallPorts}
import buckyball.balls.mxint8quant.{Params => ParametersProto}
import framework.ball.blink.{BlinkIO, HasBlink}

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}

/**
 * MXINT8QUANT: rows of K FP32 values to packed MXINT8 rows. The input streams one bank row a cycle into two slots of
 * one group each: a block of a row (eight bank rows), or with columns a block of four rows (32 bank rows, one element
 * of each of the four rows a bank row). The input rows are inPitch bank rows apart: with columns they are the rows of
 * a row-major matrix [K][rows], read four columns (a bank row of each) at a time. While the next group is read, the group of the other slot is written: two
 * code rows for each of its rows, and at every 16th or a row's last block that row's zero-padded exponent row.
 */
@instantiable
class Mxint8quantBall(
  link:  BlinkParam,
  route: BallRouteParam)
    extends Module
    with HasBlink {
  val ball = route

  ball.parameters.unpack(classOf[ParametersProto])

  require(link.bankWidth == 128 && link.bankMaskLen == 16, "Mxint8quantBall requires 128-bit bank rows and byte masks")
  require(ball.inBW == 1 && ball.outBW == 1, "Mxint8quantBall requires one read and one write")
  require(link.dataBits == 64, "Mxint8quantBall decodes a 64-bit rs2")

  @public
  val blink = IO(new BlinkIO(link, route))

  val busy        = RegInit(false.B)
  val done        = RegInit(false.B)
  val robId       = Reg(UInt(link.robIdBits.W))
  val source      = Reg(UInt(link.bankIdBits.W))
  val destination = Reg(UInt(link.bankIdBits.W))
  val sourceGroup = Reg(UInt(link.groupIdBits.W))
  val targetGroup = Reg(UInt(link.groupIdBits.W))
  val columns     = Reg(Bool())
  val k           = Reg(UInt(16.W))
  val stride      = Reg(UInt(16.W))
  val inBase      = Reg(UInt(14.W))
  val inPitch     = Reg(UInt(14.W))
  // The bank row of the next read, its place in its input row (or, with columns, its four columns' K rows) and, with
  // columns, the four columns.
  val address     = Reg(UInt(14.W))
  val line        = Reg(UInt(16.W))
  val quad        = Reg(UInt(14.W))
  val outBase     = Reg(UInt(14.W))
  val total       = Reg(UInt(32.W))

  val active = !reset.asBool
  blink.cmdReq.ready            := !busy && !done && active
  blink.cmdResp.valid           := done && active
  blink.cmdResp.bits.rob_id     := robId
  blink.status.idle             := !busy && !done
  blink.status.running          := busy
  when(blink.cmdResp.fire)(done := false.B)

  // Rows a group (1 or 4) and bank rows a group (8 or 32).
  val lanes      = Mux(columns, 4.U, 1.U)
  val groupShift = Mux(columns, 5.U, 3.U)
  val blocks     = k >> 5

  // The input: bank row n of the group n >> groupShift goes to its slot.
  val issued   = RegInit(0.U(32.W))
  val received = RegInit(0.U(32.W))
  val drained  = RegInit(0.U(32.W))
  val full     = RegInit(VecInit(Seq.fill(2)(false.B)))
  val values   = Reg(Vec(2, Vec(128, UInt(32.W))))
  val maxima   = Reg(Vec(2, Vec(4, UInt(31.W))))

  val read = blink.bankRead(0)
  read.rob_id           := robId
  read.bank_id          := source
  read.group_id         := sourceGroup
  read.io.req.valid     := busy && issued < total && (issued >> groupShift) < drained + 2.U && active
  read.io.req.bits.addr := address
  read.io.resp.ready    := busy && active
  when(read.io.req.fire) {
    issued := issued + 1.U
    line   := line + 1.U
    when(columns) {
      address := address + inPitch
      when(line === k - 1.U) {
        line    := 0.U
        quad    := quad + 1.U
        address := inBase + quad + 1.U
      }
    }.otherwise {
      address := address + 1.U
      when(line === (k >> 2) - 1.U) {
        line    := 0.U
        address := address + inPitch - ((k >> 2) - 1.U)
      }
    }
  }

  val fill   = (received >> groupShift)(0)
  val within = received & Mux(columns, 31.U, 7.U)
  when(read.io.resp.fire) {
    val words = (0 until 4).map(i => read.io.resp.bits.data(32 * i + 31, 32 * i))
    val old   = maxima(fill).map(m => Mux(within === 0.U, 0.U, m))
    def larger(a: UInt, b: UInt) = Mux(a > b, a, b)
    for (i <- 0 until 4) {
      assert(words(i)(30, 23) =/= 255.U, "MXINT8QUANT input must be finite")
      values(fill)(Cat(within(4, 0), i.U(2.W))) := words(i)
      maxima(fill)(i)                           := Mux(
        columns,
        larger(old(i), words(i)(30, 0)),
        if (i == 0) (old(0) +: words.map(_(30, 0))).reduce(larger) else old(i)
      )
    }
    received := received + 1.U
    when(within === Mux(columns, 31.U, 7.U))(full(fill) := true.B)
  }

  // The output: group `drained`, block `block` of unit `unit` (a row, or four rows with columns).
  val slot     = drained(0)
  val block    = RegInit(0.U(16.W))
  val unit     = RegInit(0.U(16.W))
  val write    = blink.bankWrite(0)
  val writing  = busy && full(slot)
  val index    = RegInit(0.U(4.W))
  val codes    = lanes << 1
  val boundary = block(3, 0) === 15.U || block === blocks - 1.U
  val count    = codes + Mux(boundary, lanes, 0.U)
  val code     = index < codes
  val row      = Mux(code, index >> 1, index - codes)
  val half     = index(0)

  val exponents = VecInit(maxima(slot).map { m =>
    val raw = m(30, 23).zext - 133.S
    Mux(m === 0.U, 0.S(10.W), Mux(raw < -127.S, -127.S(10.W), raw))
  })

  val exponent = exponents(row(1, 0))

  // The exponent bytes of each row's current 16 blocks, the current block's from this group.
  val scales = RegInit(VecInit(Seq.fill(4)(VecInit(Seq.fill(16)(0.U(8.W))))))

  val scaleRow = VecInit(scales(row(1, 0)).zipWithIndex.map { case (byte, i) =>
    Mux(block(3, 0) === i.U, (exponent + 127.S).asUInt(7, 0), byte)
  })

  val encoders = Seq.fill(16)(Instantiate(new Encode))
  for ((encoder, i) <- encoders.zipWithIndex) {
    val element = Mux(columns, Cat(half, i.U(4.W), row(1, 0)), Cat(half, i.U(4.W)))
    encoder.io.valid    := writing && code
    encoder.io.input    := values(slot)(element)
    encoder.io.exponent := exponent
  }

  val rowBase = outBase + (unit * lanes + row) * stride
  write.rob_id           := robId
  write.bank_id          := destination
  write.group_id         := targetGroup
  write.io.req.valid     := writing && active
  write.io.req.bits.addr := Mux(code, rowBase + (block << 1) + half, rowBase + (k >> 4) + (block >> 4))
  write.io.req.bits.data := Mux(code, Cat(encoders.reverse.map(_.io.code)), Cat(scaleRow.reverse))
  write.io.req.bits.mask := VecInit(Seq.fill(16)(true.B))
  write.io.resp.ready    := active
  val outstanding = RegInit(0.U(8.W))
  when(write.io.req.fire =/= write.io.resp.fire)(outstanding := Mux(
    write.io.req.fire,
    outstanding + 1.U,
    outstanding - 1.U
  ))
  when(write.io.resp.fire)(assert(write.io.resp.bits.ok, "MXINT8QUANT bank write failed"))

  when(write.io.req.fire) {
    index := index + 1.U
    when(index === count - 1.U) {
      index      := 0.U
      full(slot) := false.B
      drained    := drained + 1.U
      for (r <- 0 until 4)
        when(r.U < lanes) {
          when(boundary)(scales(r).foreach(_ := 0.U)).otherwise {
            scales(r)(block(3, 0)) := (exponents(r) + 127.S).asUInt(7, 0)
          }
        }
      block := Mux(block === blocks - 1.U, 0.U, block + 1.U)
      when(block === blocks - 1.U)(unit := unit + 1.U)
    }
  }

  when(busy && received === total && !full(0) && !full(1) && outstanding === 0.U && !write.io.req.fire) {
    busy := false.B
    done := true.B
  }

  when(blink.cmdReq.fire) {
    val cmd      = blink.cmdReq.bits
    val rows     = cmd.iter
    val count    = cmd.rs2(15, 0)
    val packed   = (count >> 4) +& (count >> 9) +& (count(8, 0) =/= 0.U)
    val byColumn = cmd.rs2(44)
    val pitch    = cmd.rs2(58, 45)
    // The bank rows of an input row and the input rows.
    val width    = Mux(byColumn, rows >> 2, count >> 2)
    val inputs   = Mux(byColumn, count, rows)
    assert(cmd.rs2(63, 59) === 0.U, "MXINT8QUANT reserved fields must be zero")
    assert(pitch >= width, "MXINT8QUANT input rows overlap")
    assert(
      rows =/= 0.U && (!byColumn || rows(1, 0) === 0.U),
      "MXINT8QUANT rows must be positive (a multiple of 4 with columns)"
    )
    assert(count =/= 0.U && count(4, 0) === 0.U, "MXINT8QUANT K must be a positive multiple of 32")
    assert(cmd.op1_bank =/= cmd.wr_bank, "MXINT8QUANT input and output banks must differ")
    assert(cmd.op1_col === 1.U && cmd.wr_col === 1.U, "MXINT8QUANT banks must have one group")
    assert(
      cmd.rs2(29, 16) +& (inputs - 1.U) * pitch +& width <= link.bankEntries.U,
      "MXINT8QUANT input exceeds capacity"
    )
    assert(cmd.rs2(43, 30) +& rows * packed <= link.bankEntries.U, "MXINT8QUANT output exceeds capacity")
    busy        := true.B
    robId       := cmd.rob_id
    source      := cmd.op1_bank
    destination := cmd.wr_bank
    sourceGroup := cmd.op1_group
    targetGroup := cmd.wr_group
    columns     := byColumn
    k           := count
    stride      := packed
    inBase      := cmd.rs2(29, 16)
    inPitch     := pitch
    address     := cmd.rs2(29, 16)
    line        := 0.U
    quad        := 0.U
    outBase     := cmd.rs2(43, 30)
    total       := rows * (count >> 2)
    issued      := 0.U
    received    := 0.U
    drained     := 0.U
    block       := 0.U
    unit        := 0.U
    index       := 0.U
    scales.foreach(_.foreach(_ := 0.U))
  }
}

object Mxint8quantBall extends BallFactory {

  def instantiate(
    link:      BlinkParam,
    route:     BallRouteParam,
    expansion: Option[ExpansionParam]
  ): BallPorts =
    BallPorts(Instantiate(new Mxint8quantBall(link, route)).blink, None)

}
