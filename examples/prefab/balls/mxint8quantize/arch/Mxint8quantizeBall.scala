package examples.balls.mxint8quantize

import framework.acc.balldomain.bbus.{BallFactory, BallPorts}
import buckyball.balls.mxint8quantize.{Params => ParametersProto}
import framework.acc.balldomain.configs.BallDomainParam
import framework.acc.frontend.configs.FrontendParam
import framework.acc.memdomain.configs.MemDomainParam
import framework.acc.balldomain.ball.blink.{BlinkIO, HasBlink}

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}

/**
 * MXINT8QUANT: rows of K FP32 values to packed MXINT8 rows. One block of 32 values at a time: eight bank reads, the
 * block exponent, then two code rows and, at every 16th or the row's last block, the zero-padded exponent row.
 */
@instantiable
class Mxint8quantizeBall(
  fp: FrontendParam,
  mp: MemDomainParam,
  bp: BallDomainParam,
  id: Int)
    extends Module
    with HasBlink {
  val ball = bp.balls.find(_.id == id).get

  ball.parameters.unpack(classOf[ParametersProto])

  require(mp.bankWidth == 128 && mp.bankMaskLen == 16, "Mxint8quantizeBall requires 128-bit bank rows and byte masks")
  require(ball.inBW == 1 && ball.outBW == 1, "Mxint8quantizeBall requires one read and one write")
  require(fp.commandDataBits == 64, "Mxint8quantizeBall decodes a 64-bit rs2")

  @public
  val blink = IO(new BlinkIO(fp, mp, bp, id))

  val idle :: reading :: exponent :: writing :: complete :: Nil = Enum(5)
  val state                                                     = RegInit(idle)

  val robId       = Reg(UInt(log2Up(fp.rob_entries).W))
  val source      = Reg(UInt(fp.bank_id_len.W))
  val destination = Reg(UInt(fp.bank_id_len.W))
  val sourceGroup = Reg(UInt(mp.groupIdWidth.W))
  val targetGroup = Reg(UInt(mp.groupIdWidth.W))
  val rows        = Reg(UInt(fp.iter_len.W))
  val k           = Reg(UInt(16.W))
  val stride      = Reg(UInt(16.W))
  val inAddress   = Reg(UInt(32.W))
  val rowBase     = Reg(UInt(32.W))
  val row         = Reg(UInt(fp.iter_len.W))
  val block       = Reg(UInt(16.W))
  val readIssued  = Reg(UInt(4.W))
  val readDone    = Reg(UInt(4.W))
  val writeIssued = Reg(UInt(2.W))
  val writeDone   = Reg(UInt(2.W))
  val maximum     = Reg(UInt(31.W))
  val scale       = Reg(SInt(10.W))
  val values      = Reg(Vec(32, UInt(32.W)))
  val scaleRow    = RegInit(VecInit(Seq.fill(16)(0.U(8.W))))

  val lastBlock  = block === (k >> 5) - 1.U
  val scaleWrite = block(3, 0) === 15.U || lastBlock
  val writeCount = Mux(scaleWrite, 3.U, 2.U)

  val encoders = Seq.fill(16)(Instantiate(new Encode))
  for ((encoder, i) <- encoders.zipWithIndex) {
    encoder.io.valid    := state === writing && writeIssued < 2.U
    encoder.io.input    := Mux(writeIssued(0), values(16 + i), values(i))
    encoder.io.exponent := scale
  }

  val active = !reset.asBool
  blink.cmdReq.ready        := state === idle && active
  blink.cmdResp.valid       := state === complete && active
  blink.cmdResp.bits.rob_id := robId
  blink.status.idle         := state === idle
  blink.status.running      := state =/= idle && state =/= complete

  val read = blink.bankRead(0)
  read.rob_id           := robId
  read.bank_id          := source
  read.group_id         := sourceGroup
  read.io.req.valid     := state === reading && readIssued < 8.U && active
  read.io.req.bits.addr := inAddress + readIssued
  read.io.resp.ready    := state === reading && active

  val write = blink.bankWrite(0)
  write.rob_id           := robId
  write.bank_id          := destination
  write.group_id         := targetGroup
  write.io.req.valid     := state === writing && writeIssued < writeCount && active
  write.io.req.bits.addr := Mux(
    writeIssued === 2.U,
    rowBase + (k >> 4) + (block >> 4),
    rowBase + (block << 1) + writeIssued
  )
  write.io.req.bits.data := Mux(
    writeIssued === 2.U,
    Cat(scaleRow.reverse),
    Cat(encoders.reverse.map(_.io.code))
  )
  write.io.req.bits.mask := VecInit(Seq.fill(16)(true.B))
  write.io.resp.ready    := state === writing && active

  when(read.io.req.fire)(readIssued   := readIssued + 1.U)
  when(write.io.req.fire)(writeIssued := writeIssued + 1.U)
  when(write.io.resp.fire) {
    assert(write.io.resp.bits.ok, "MXINT8QUANT bank write failed")
    writeDone := writeDone + 1.U
  }

  switch(state) {
    is(idle) {
      when(blink.cmdReq.fire) {
        val cmd     = blink.cmdReq.bits
        val count   = cmd.rs2(15, 0)
        val inBase  = cmd.rs2(29, 16)
        val outBase = cmd.rs2(43, 30)
        val packed  = (count >> 4) +& (count >> 9) +& (count(8, 0) =/= 0.U)
        assert(cmd.rs2(63, 44) === 0.U, "MXINT8QUANT reserved fields must be zero")
        assert(cmd.iter =/= 0.U, "MXINT8QUANT rows must be positive")
        assert(count =/= 0.U && count(4, 0) === 0.U, "MXINT8QUANT K must be a positive multiple of 32")
        assert(cmd.op1_bank =/= cmd.wr_bank, "MXINT8QUANT input and output banks must differ")
        assert(cmd.op1_col === 1.U && cmd.wr_col === 1.U, "MXINT8QUANT banks must have one group")
        assert(inBase +& cmd.iter * (count >> 2) <= mp.bankEntries.U, "MXINT8QUANT input exceeds capacity")
        assert(outBase +& cmd.iter * packed <= mp.bankEntries.U, "MXINT8QUANT output exceeds capacity")
        robId       := cmd.rob_id
        source      := cmd.op1_bank
        destination := cmd.wr_bank
        sourceGroup := cmd.op1_group
        targetGroup := cmd.wr_group
        rows        := cmd.iter
        k           := count
        stride      := packed
        inAddress   := inBase
        rowBase     := outBase
        row         := 0.U
        block       := 0.U
        readIssued  := 0.U
        readDone    := 0.U
        maximum     := 0.U
        scaleRow    := VecInit(Seq.fill(16)(0.U(8.W)))
        state       := reading
      }
    }
    is(reading) {
      when(read.io.resp.fire) {
        val words = (0 until 4).map(i => read.io.resp.bits.data(32 * i + 31, 32 * i))
        for (i <- 0 until 4) {
          assert(words(i)(30, 23) =/= 255.U, "MXINT8QUANT input must be finite")
          values(Cat(readDone(2, 0), i.U(2.W))) := words(i)
        }
        maximum := (Seq(maximum) ++ words.map(_(30, 0))).reduce((a, b) => Mux(a > b, a, b))
        readDone                     := readDone + 1.U
        when(readDone === 7.U)(state := exponent)
      }
    }
    is(exponent) {
      val raw      = maximum(30, 23).zext - 133.S
      val selected = Mux(maximum === 0.U, 0.S(10.W), Mux(raw < -127.S, -127.S(10.W), raw))
      scale                 := selected
      scaleRow(block(3, 0)) := (selected + 127.S).asUInt(7, 0)
      writeIssued           := 0.U
      writeDone             := 0.U
      state                 := writing
    }
    is(writing) {
      when(write.io.resp.fire && writeDone === writeCount - 1.U) {
        when(scaleWrite)(scaleRow := VecInit(Seq.fill(16)(0.U(8.W))))
        inAddress                 := inAddress + 8.U
        readIssued                := 0.U
        readDone                  := 0.U
        maximum                   := 0.U
        state                     := reading
        when(lastBlock) {
          block                          := 0.U
          row                            := row + 1.U
          rowBase                        := rowBase + stride
          when(row === rows - 1.U)(state := complete)
        }.otherwise(block         := block + 1.U)
      }
    }
    is(complete)(when(blink.cmdResp.fire)(state := idle))
  }
}

object Mxint8quantizeBall extends BallFactory {

  def instantiate(
    fp: FrontendParam,
    mp: MemDomainParam,
    bp: BallDomainParam,
    id: Int
  ): BallPorts =
    BallPorts(Instantiate(new Mxint8quantizeBall(fp, mp, bp, id)).blink, None)

}
