package examples.balls.maxpool

import framework.generic.contract.blink.BlinkParam
import framework.generic.contract.configs.BallRouteParam
import framework.ball.sub_rob.ExpansionParam

import framework.ball.bbus.{BallFactory, BallPorts}
import buckyball.balls.maxpool.{Params => ParametersProto}

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import chisel3.util._
import framework.ball.blink.{BallStatus, BlinkIO, HasBlink}

@instantiable
class MaxPoolBall(
  link:  BlinkParam,
  route: BallRouteParam)
    extends Module
    with HasBlink {
  val ball = route

  ball.parameters.unpack(classOf[ParametersProto])
  val maxpoolFunct = ball.instructions.find(_.mnemonic == "MAXPOOL").get.funct7
  val topkFunct    = ball.instructions.find(_.mnemonic == "TOPK_U32").get.funct7
  require((maxpoolFunct >> 4) == 3 && (topkFunct >> 4) == 3, "MaxPoolBall instructions require one read and one write")

  require(link.bankWidth == 128, "MaxPoolBall requires 128-bit bank rows")
  require(ball.inBW == 1, "MaxPoolBall requires inBW=1")
  require(ball.outBW == 1, "MaxPoolBall requires outBW=1")

  @public
  val blink = IO(new BlinkIO(link, route))

  val idle :: scan :: readResp :: writeReq :: writeResp :: complete :: topScan :: topWriteReq :: topWriteResp :: Nil =
    Enum(9)
  val state                                                                                                          = RegInit(idle)
  val robId                                                                                                          = RegInit(0.U(link.robIdBits.W))
  val inputBank                                                                                                      = RegInit(0.U(link.bankIdBits.W))
  val outputBank                                                                                                     = RegInit(0.U(link.bankIdBits.W))
  val bases                                                                                                          = Reg(Vec(3, UInt(link.groupIdBits.W)))
  val iter                                                                                                           = RegInit(0.U(link.iterBits.W))
  // Every group of a virtual bank is one 16-channel block with the same rows.
  val groups                                                                                                         = RegInit(0.U(log2Up(link.bankNum + 1).W))
  val group                                                                                                          = RegInit(0.U(log2Up(link.bankNum + 1).W))
  val inputSide                                                                                                      = RegInit(0.U(4.W))
  val outputSide                                                                                                     = RegInit(0.U(4.W))
  val kernel                                                                                                         = RegInit(0.U(4.W))
  val stride                                                                                                         = RegInit(0.U(4.W))
  val padding                                                                                                        = RegInit(0.U(4.W))
  val inputBase                                                                                                      = RegInit(0.U(6.W))
  val outputBase                                                                                                     = RegInit(0.U(6.W))
  val outputStride                                                                                                   = RegInit(0.U(6.W))
  val startRow                                                                                                       = RegInit(0.U(4.W))
  val startCol                                                                                                       = RegInit(0.U(4.W))
  val outputRow                                                                                                      = RegInit(0.U(log2Ceil(link.bankEntries).W))
  val outputY                                                                                                        = RegInit(0.U(4.W))
  val outputX                                                                                                        = RegInit(0.U(4.W))
  val kernelY                                                                                                        = RegInit(0.U(4.W))
  val kernelX                                                                                                        = RegInit(0.U(4.W))
  val maximum                                                                                                        = RegInit(VecInit(Seq.fill(16)("h80".U(8.W))))
  val writeData                                                                                                      = Reg(UInt(128.W))

  val topIndexBits   = log2Ceil(link.bankEntries * link.bankNum * 4)
  val topKeyBits     = topIndexBits + 33
  val topIndexBase   = RegInit(0.U(topIndexBits.W))
  val topCount       = RegInit(0.U(32.W))
  val topK           = RegInit(0.U(16.W))
  val topRank        = RegInit(0.U(16.W))
  val topRow         = RegInit(0.U(log2Up(link.bankEntries).W))
  val topRequested   = RegInit(0.U(log2Up(link.bankEntries + 1).W))
  val topGroup       = RegInit(0.U(log2Up(link.bankNum + 1).W))
  val topBest        = RegInit(VecInit(Seq.fill(4)(0.U(topKeyBits.W))))
  val topPrevious    = RegInit(0.U(topKeyBits.W))
  val topHasPrevious = RegInit(false.B)
  val topOutputHigh  = Reg(UInt(128.W))
  val topWriteHalf   = RegInit(false.B)
  val topRowReady    = state === topScan && blink.bankRead(0).io.resp.fire
  val topRowValid    = RegNext(topRowReady, false.B)
  val topInput       = Wire(Vec(4, UInt(topKeyBits.W)))
  for (lane  <- 0 until 4) {
    val at     = Cat(topIndexBase(topIndexBits - 1, 2), lane.U(2.W))
    val key    = blink.bankRead(0).io.resp.bits.data(32 * lane + 31, 32 * lane)
    val ranked = Cat(1.U(1.W), key, ~at)
    topInput(lane) := Mux(at < topCount && (!topHasPrevious || ranked < topPrevious), ranked, 0.U)
  }
  var sortedRow: Seq[UInt] = topInput.toSeq
  for (pairs <- Seq(Seq((0, 1), (2, 3)), Seq((0, 2), (1, 3)), Seq((1, 2)))) {
    val next = Wire(Vec(4, UInt(topKeyBits.W)))
    next := VecInit(sortedRow)
    for ((left, right) <- pairs) {
      next(left)  := Mux(sortedRow(left) > sortedRow(right), sortedRow(left), sortedRow(right))
      next(right) := Mux(sortedRow(left) > sortedRow(right), sortedRow(right), sortedRow(left))
    }
    sortedRow = next.toSeq
  }
  val topSortedRow = RegEnable(VecInit(sortedRow), topRowReady)
  var merged: Seq[UInt] = topSortedRow.toSeq ++ topBest.reverse
  for (distance <- Seq(4, 2, 1)) {
    val next = Wire(Vec(8, UInt(topKeyBits.W)))
    for (lane <- 0 until 8) {
      val peer    = lane ^ distance
      val greater = merged(lane) > merged(peer)
      next(lane) := Mux(if ((lane & distance) == 0) greater else !greater, merged(lane), merged(peer))
    }
    merged = next.toSeq
  }

  val sourceY       = outputY * stride +& kernelY +& startRow
  val sourceX       = outputX * stride +& kernelX +& startCol
  val sourceValid   = sourceY >= padding && sourceX >= padding &&
    sourceY < padding +& inputSide && sourceX < padding +& inputSide
  val inputY        = (sourceY - padding)(3, 0)
  val inputX        = (sourceX - padding)(3, 0)
  val addressBits   = log2Up(link.bankEntries)
  val inputAddress  = (inputBase +& inputY * inputSide +& inputX).pad(addressBits)(addressBits - 1, 0)
  val outputAddress = (outputBase +& outputY * outputStride +& outputX).pad(addressBits)(addressBits - 1, 0)
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
    Seq.fill(link.bankMaskLen)(true.B)
  )
  blink.bankWrite(0).io.resp.ready    := false.B

  switch(state) {
    is(idle) {
      when(blink.cmdReq.fire) {
        val cmd = blink.cmdReq.bits
        when(cmd.funct === topkFunct.U) {
          val n = cmd.rs2(31, 0)
          val k = cmd.rs2(47, 32)
          assert(cmd.rs2(63, 48) === 0.U, "TOPK_U32 reserved rs2 bits must be zero")
          assert(
            cmd.op1_col > 0.U && cmd.op1_group +& cmd.op1_col <= link.bankNum.U &&
              cmd.op2_col === 0.U && cmd.wr_col === 1.U && cmd.wr_group < link.bankNum.U,
            "TOPK_U32 requires input groups and one output group"
          )
          assert(cmd.op1_bank =/= cmd.wr_bank, "TOPK_U32 banks must differ")
          assert(
            cmd.iter > 0.U && cmd.iter <= link.bankEntries.U && n > 0.U &&
              n <= ((cmd.iter * cmd.op1_col) << 2),
            "TOPK_U32 input exceeds bank geometry"
          )
          assert(
            k > 0.U && k <= n && k <= (2 * link.bankEntries).U,
            "TOPK_U32 result count exceeds input or output capacity"
          )
          robId          := cmd.rob_id
          inputBank      := cmd.op1_bank
          outputBank     := cmd.wr_bank
          bases          := VecInit(cmd.op1_group, cmd.op2_group, cmd.wr_group)
          iter           := cmd.iter
          groups         := cmd.op1_col
          group          := 0.U
          topIndexBase   := 0.U
          topCount       := n
          topK           := k
          topRank        := 0.U
          topRequested   := 0.U
          topRow         := 0.U
          topGroup       := 0.U
          topBest        := VecInit(Seq.fill(4)(0.U(topKeyBits.W)))
          topHasPrevious := false.B
          topWriteHalf   := false.B
          state          := topScan
        }.otherwise {
          assert(cmd.funct === maxpoolFunct.U, "MaxPoolBall received an unknown funct7")
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
              cmd.op1_col <= link.bankNum.U,
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
            !readsInput || inBase +& maxInputRow * inSide +& maxInputCol < link.bankEntries.U,
            "MAXPOOL input window exceeds bank depth"
          )
          assert(
            outSide * outSide === cmd.iter &&
              outBase +& (outSide - 1.U) * outStride +& outSide <= link.bankEntries.U,
            "MAXPOOL output hart exceeds bank depth"
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

    is(topScan) {
      blink.bankRead(0).group_id         := bases(0) + topGroup
      blink.bankRead(0).io.req.bits.addr := topRequested(addressBits - 1, 0)
      blink.bankRead(0).io.req.valid     := topRequested < iter
      blink.bankRead(0).io.resp.ready    := true.B
      when(blink.bankRead(0).io.req.fire) {
        topRequested := topRequested + 1.U
      }
      when(topRowReady) {
        topIndexBase := topIndexBase + (groups << 2)
      }
      when(topRowValid) {
        topBest := VecInit(merged.take(4))
        when(topRow +& 1.U === iter && topGroup +& 1.U === groups) {
          val pairs = Wire(Vec(4, UInt(64.W)))
          for (rank <- 0 until 4) {
            val selected = topRank +& rank.U < topK
            when(selected)(assert(merged(rank)(topKeyBits - 1), "TOPK_U32 found no remaining key"))
            pairs(rank) := Mux(
              selected,
              Cat((~merged(rank)(topIndexBits - 1, 0)).pad(32), merged(rank)(topIndexBits + 31, topIndexBits)),
              0.U
            )
          }
          writeData := Cat(pairs(1), pairs(0))
          topOutputHigh  := Cat(pairs(3), pairs(2))
          topPrevious    := merged(3)
          topHasPrevious := true.B
          topBest        := VecInit(Seq.fill(4)(0.U(topKeyBits.W)))
          topIndexBase   := 0.U
          topRequested   := 0.U
          topRow         := 0.U
          topGroup       := 0.U
          topWriteHalf   := false.B
          state          := topWriteReq
        }.elsewhen(topRow +& 1.U === iter) {
          topIndexBase := (topGroup +& 1.U) << 2
          topRequested := 0.U
          topRow       := 0.U
          topGroup     := topGroup + 1.U
        }.otherwise {
          topRow := topRow + 1.U
        }
      }
    }

    is(topWriteReq) {
      blink.bankWrite(0).group_id         := bases(2)
      blink.bankWrite(0).io.req.bits.addr := (topRank >> 1) + topWriteHalf
      blink.bankWrite(0).io.req.valid     := true.B
      when(blink.bankWrite(0).io.req.fire) {
        state := topWriteResp
      }
    }

    is(topWriteResp) {
      blink.bankWrite(0).group_id      := bases(2)
      blink.bankWrite(0).io.resp.ready := true.B
      when(blink.bankWrite(0).io.resp.fire) {
        when(!topWriteHalf && topRank +& 2.U < topK) {
          writeData    := topOutputHigh
          topWriteHalf := true.B
          state        := topWriteReq
        }.elsewhen(topRank +& 4.U >= topK) {
          state := complete
        }.otherwise {
          topRank := topRank + 4.U
          state   := topScan
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
    link:      BlinkParam,
    route:     BallRouteParam,
    expansion: Option[ExpansionParam]
  ): BallPorts =
    BallPorts(Instantiate(new MaxPoolBall(link, route)).blink, None)

}
