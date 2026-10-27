package examples.balls.maxpool

import framework.acc.balldomain.bbus.{BallFactory, BallPorts}
import buckyball.balls.maxpool.{Params => ParametersProto}

import framework.acc.balldomain.configs.BallDomainParam
import framework.acc.frontend.configs.FrontendParam
import framework.acc.memdomain.configs.MemDomainParam

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import chisel3.util._
import framework.acc.balldomain.ball.blink.{BallStatus, BlinkIO, HasBlink}

@instantiable
class MaxPoolBall(
  fp: FrontendParam,
  mp: MemDomainParam,
  bp: BallDomainParam,
  id: Int)
    extends Module
    with HasBlink {
  val ball = bp.balls.find(_.id == id).get

  ball.parameters.unpack(classOf[ParametersProto])

  require(mp.bankWidth == 128, "MaxPoolBall requires 128-bit bank rows")
  require(ball.inBW == 1, "MaxPoolBall requires inBW=1")
  require(ball.outBW == 1, "MaxPoolBall requires outBW=1")

  @public
  val blink = IO(new BlinkIO(fp, mp, bp, id))

  val idle :: scan :: readResp :: writeReq :: writeResp :: complete :: Nil = Enum(6)
  val state                                                                = RegInit(idle)
  val robId                                                                = RegInit(0.U(log2Up(fp.rob_entries).W))
  val inputBank                                                            = RegInit(0.U(fp.bank_id_len.W))
  val outputBank                                                           = RegInit(0.U(fp.bank_id_len.W))
  val bases                                                                = Reg(Vec(3, UInt(mp.groupIdWidth.W)))
  val iter                                                                 = RegInit(0.U(fp.iter_len.W))
  // Every group of a virtual bank is one 16-channel block with the same rows.
  val groups                                                               = RegInit(0.U(log2Up(mp.bankNum + 1).W))
  val group                                                                = RegInit(0.U(log2Up(mp.bankNum + 1).W))
  val inputSide                                                            = RegInit(0.U(8.W))
  val outputSide                                                           = RegInit(0.U(8.W))
  val kernel                                                               = RegInit(0.U(8.W))
  val stride                                                               = RegInit(0.U(8.W))
  val padding                                                              = RegInit(0.U(8.W))
  val inputBase                                                            = RegInit(0.U(6.W))
  val outputBase                                                           = RegInit(0.U(6.W))
  val outputStride                                                         = RegInit(0.U(6.W))
  val startRow                                                             = RegInit(0.U(4.W))
  val startCol                                                             = RegInit(0.U(4.W))
  val outputRow                                                            = RegInit(0.U(log2Ceil(mp.bankEntries).W))
  val outputY                                                              = RegInit(0.U(8.W))
  val outputX                                                              = RegInit(0.U(8.W))
  val kernelY                                                              = RegInit(0.U(8.W))
  val kernelX                                                              = RegInit(0.U(8.W))
  val maximum                                                              = RegInit(VecInit(Seq.fill(16)("h80".U(8.W))))
  val writeData                                                            = Reg(UInt(128.W))

  val sourceY       = outputY * stride +& kernelY +& startRow
  val sourceX       = outputX * stride +& kernelX +& startCol
  val sourceValid   = sourceY >= padding && sourceX >= padding &&
    sourceY < padding +& inputSide && sourceX < padding +& inputSide
  val inputY        = sourceY - padding
  val inputX        = sourceX - padding
  val inputAddress  = inputBase +& inputY * inputSide +& inputX
  val outputAddress = outputBase +& outputY * outputStride +& outputX
  val lastKernel    = kernelY +& 1.U === kernel && kernelX +& 1.U === kernel

  blink.cmdReq.ready        := state === idle
  blink.cmdResp.valid       := state === complete
  blink.cmdResp.bits.rob_id := robId
  blink.status.idle         := state === idle
  blink.status.running      := state =/= idle && state =/= complete

  blink.bankRead(0).rob_id           := robId
  blink.bankRead(0).bank_id          := inputBank
  blink.bankRead(0).group_id         := bases(0) + group
  blink.bankRead(0).io.req.valid     := false.B
  blink.bankRead(0).io.req.bits.addr := inputAddress
  blink.bankRead(0).io.resp.ready    := false.B

  blink.bankWrite(0).rob_id           := robId
  blink.bankWrite(0).bank_id          := outputBank
  blink.bankWrite(0).group_id         := bases(2) + group
  blink.bankWrite(0).io.req.valid     := false.B
  blink.bankWrite(0).io.req.bits.addr := outputAddress
  blink.bankWrite(0).io.req.bits.data := writeData
  blink.bankWrite(0).io.req.bits.mask := VecInit(
    Seq.fill(mp.bankMaskLen)(true.B)
  )
  blink.bankWrite(0).io.resp.ready    := false.B

  switch(state) {
    is(idle) {
      when(blink.cmdReq.fire) {
        val cmd          = blink.cmdReq.bits
        val inSide       = cmd.rs2(3, 0)
        val outSide      = cmd.rs2(7, 4)
        val kernelSize   = cmd.rs2(11, 8)
        val poolStride   = cmd.rs2(15, 12)
        val poolPadding  = cmd.rs2(19, 16)
        val inBase       = cmd.rs2(25, 20)
        val outBase      = cmd.rs2(31, 26)
        val outStride    = cmd.rs2(37, 32)
        val poolStartRow = cmd.rs2(41, 38)
        val poolStartCol = cmd.rs2(45, 42)
        assert(cmd.rs2(63, 46) === 0.U, "MAXPOOL reserved rs2 bits must be zero")

        assert(
          cmd.op1_col =/= 0.U && cmd.op2_col === 0.U && cmd.wr_col === cmd.op1_col &&
            cmd.op1_col <= mp.bankNum.U,
          "MAXPOOL input and output must have the same channel groups"
        )
        assert(cmd.op1_bank =/= cmd.wr_bank, "MAXPOOL input and output banks must differ")
        assert(
          inSide > 0.U && outSide > 0.U && kernelSize > 0.U && poolStride > 0.U,
          "MAXPOOL dimensions must be positive"
        )
        assert(outStride >= outSide, "MAXPOOL output stride is too small")
        val lastSourceRow = poolStartRow +& (outSide - 1.U) * poolStride +& (kernelSize - 1.U)
        val lastSourceCol = poolStartCol +& (outSide - 1.U) * poolStride +& (kernelSize - 1.U)
        val inputEnd      = poolPadding +& inSide
        val readsInput    = poolStartRow < inputEnd && poolStartCol < inputEnd &&
          lastSourceRow >= poolPadding && lastSourceCol >= poolPadding
        val maxInputRow   = Mux(lastSourceRow < inputEnd, lastSourceRow - poolPadding, inSide - 1.U)
        val maxInputCol   = Mux(lastSourceCol < inputEnd, lastSourceCol - poolPadding, inSide - 1.U)
        assert(
          !readsInput || inBase +& maxInputRow * inSide +& maxInputCol < mp.bankEntries.U,
          "MAXPOOL input window exceeds bank depth"
        )
        assert(
          outSide * outSide === cmd.iter &&
            outBase +& (outSide - 1.U) * outStride +& outSide <= mp.bankEntries.U,
          "MAXPOOL output core exceeds bank depth"
        )
        assert(
          poolStartRow +& (outSide - 1.U) * poolStride +& kernelSize <=
            inSide +& (poolPadding << 1) &&
            poolStartCol +& (outSide - 1.U) * poolStride +& kernelSize <=
            inSide +& (poolPadding << 1),
          "MAXPOOL pooling geometry is inconsistent"
        )

        robId        := blink.cmdReq.bits.rob_id
        inputBank    := cmd.op1_bank
        outputBank   := cmd.wr_bank
        bases        := VecInit(cmd.op1_group, cmd.op2_group, cmd.wr_group)
        iter         := cmd.iter
        groups       := cmd.op1_col
        group        := 0.U
        inputSide    := inSide
        outputSide   := outSide
        kernel       := kernelSize
        stride       := poolStride
        padding      := poolPadding
        inputBase    := inBase
        outputBase   := outBase
        outputStride := outStride
        startRow     := poolStartRow
        startCol     := poolStartCol
        outputRow    := 0.U
        outputY      := 0.U
        outputX      := 0.U
        kernelY      := 0.U
        kernelX      := 0.U
        maximum      := VecInit(Seq.fill(16)("h80".U(8.W)))
        state        := scan
      }
    }

    is(scan) {
      when(sourceValid) {
        blink.bankRead(0).io.req.valid := true.B
        when(blink.bankRead(0).io.req.fire) {
          state := readResp
        }
      }.elsewhen(lastKernel) {
        writeData := Cat(maximum.reverse)
        state     := writeReq
      }.otherwise {
        when(kernelX +& 1.U === kernel) {
          kernelX := 0.U
          kernelY := kernelY + 1.U
        }.otherwise {
          kernelX := kernelX + 1.U
        }
      }
    }

    is(readResp) {
      blink.bankRead(0).io.resp.ready := true.B
      when(blink.bankRead(0).io.resp.fire) {
        val data        = blink.bankRead(0).io.resp.bits.data.asTypeOf(Vec(16, UInt(8.W)))
        val nextMaximum = Wire(Vec(16, UInt(8.W)))
        for (lane <- 0 until 16)
          nextMaximum(lane) := Mux(data(lane).asSInt > maximum(lane).asSInt, data(lane), maximum(lane))
        maximum             := nextMaximum
        when(lastKernel) {
          writeData := Cat(nextMaximum.reverse)
          state     := writeReq
        }.otherwise {
          when(kernelX +& 1.U === kernel) {
            kernelX := 0.U
            kernelY := kernelY + 1.U
          }.otherwise {
            kernelX := kernelX + 1.U
          }
          state := scan
        }
      }
    }

    is(writeReq) {
      blink.bankWrite(0).io.req.valid := true.B
      when(blink.bankWrite(0).io.req.fire) {
        state := writeResp
      }
    }

    is(writeResp) {
      blink.bankWrite(0).io.resp.ready := true.B
      when(blink.bankWrite(0).io.resp.fire) {
        when(outputRow +& 1.U === iter && group +& 1.U === groups) {
          state := complete
        }.otherwise {
          when(outputRow +& 1.U === iter) {
            group     := group + 1.U
            outputRow := 0.U
            outputY   := 0.U
            outputX   := 0.U
          }.otherwise {
            outputRow := outputRow + 1.U
            when(outputX +& 1.U === outputSide) {
              outputX := 0.U
              outputY := outputY + 1.U
            }.otherwise {
              outputX := outputX + 1.U
            }
          }
          kernelY := 0.U
          kernelX := 0.U
          maximum := VecInit(Seq.fill(16)("h80".U(8.W)))
          state   := scan
        }
      }
    }

    is(complete) {
      when(blink.cmdResp.fire) {
        state := idle
      }
    }
  }
}

object MaxPoolBall extends BallFactory {

  def instantiate(
    fp: FrontendParam,
    mp: MemDomainParam,
    bp: BallDomainParam,
    id: Int
  ): BallPorts =
    BallPorts(Instantiate(new MaxPoolBall(fp, mp, bp, id)).blink, None)

}
