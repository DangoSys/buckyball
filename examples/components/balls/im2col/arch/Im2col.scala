package examples.balls.im2col

import framework.acc.balldomain.configs.BallRouteParam

import framework.acc.balldomain.configs.BallDomainParam
import framework.acc.frontend.configs.FrontendParam
import framework.acc.memdomain.configs.MemDomainParam

import chisel3._
import framework.acc.balldomain.ball.blink.{BallCommand, BallResponse}
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import framework.acc.balldomain.ball.blink.BallStatus
import framework.acc.memdomain.{BankRead, BankWrite}
import examples.balls.im2col.configs.Im2colBallParam

/**
 * Im2col over channel-block groups.
 *
 * Input group g is channel block g; line inputBase + y * size + x is pixel
 * (y, x). K is ordered (kernel row, kernel column, channel), so K tile
 * t = (kr * kernel + kc) * inputGroups + g is one whole input row (zero in
 * the padding). Output group j holds K tiles j * groupTiles .. in the SMatMul
 * A layout: line ((w / tileRows) * groupTiles + localTile) * tileRows +
 * w % tileRows. Windows beyond windowCount in the last M tile are zero.
 */
@instantiable
class Im2col(
  val frontendParam: FrontendParam,
  val memParam:      MemDomainParam,
  val routingParam:  BallDomainParam,
  val instance:      BallRouteParam,
  val ballParam:     Im2colBallParam)
    extends Module {
  val maxIter      = ballParam.maxIter
  val maxKSize     = ballParam.maxKSize
  val maxPad       = ballParam.maxPadding
  val addressWidth = log2Ceil(memParam.bankEntries)
  val groupWidth   = log2Up(memParam.bankNum + 1)

  require(instance.inBW == 1, "Im2colBall requires inBW=1")
  require(instance.outBW == 1, "Im2colBall requires outBW=1")
  require(memParam.bankWidth == 128, "Im2colBall requires 128-bit bank rows")
  require(ballParam.inputWidth == 8, "Im2colBall moves 16 INT8 channels per row")
  require(isPow2(memParam.bankEntries), "Im2colBall requires power-of-two bank depth")
  require(maxIter + 2 * maxPad <= 255, "Im2colBall dimensions must fit the divider")
  require(maxKSize <= 15 && maxPad <= 15, "Im2colBall kernel and padding fields are four bits")

  @public
  val io = IO(new Bundle {
    val cmdReq    = Flipped(Decoupled(new BallCommand(frontendParam, memParam)))
    val cmdResp   = Decoupled(new BallResponse(frontendParam))
    val bankRead  = Vec(1, Flipped(new BankRead(frontendParam, memParam)))
    val bankWrite = Vec(1, Flipped(new BankWrite(frontendParam, memParam)))
    val status    = new BallStatus
  })

  val idle :: divRows :: divCols :: divStart :: divTiles :: check :: select :: readReq :: readResp :: writeReq :: writeResp :: complete :: Nil =
    Enum(12)
  val state                                                                                                                                    = RegInit(idle)

  val robId       = RegInit(0.U(log2Up(frontendParam.rob_entries).W))
  val inBank      = RegInit(0.U(frontendParam.bank_id_len.W))
  val outBank     = RegInit(0.U(frontendParam.bank_id_len.W))
  val inBase      = RegInit(0.U(memParam.groupIdWidth.W))
  val outBase     = RegInit(0.U(memParam.groupIdWidth.W))
  val inGroups    = RegInit(0.U(groupWidth.W))
  val outGroups   = RegInit(0.U(groupWidth.W))
  val size        = RegInit(0.U(8.W))
  val kernel      = RegInit(0.U(4.W))
  val stride      = RegInit(0.U(4.W))
  val padding     = RegInit(0.U(4.W))
  val startCol    = RegInit(0.U(4.W))
  val startRow    = RegInit(0.U(4.W))
  val inputBase   = RegInit(0.U(6.W))
  val windowStart = RegInit(0.U(10.W))
  val windowCount = RegInit(0.U(10.W))
  val tileShift   = RegInit(0.U(3.W))
  val outRows     = RegInit(0.U(16.W))
  val outCols     = RegInit(0.U(16.W))
  val firstRow    = RegInit(0.U(16.W))
  val firstCol    = RegInit(0.U(16.W))
  val groupTiles  = RegInit(0.U(16.W))

  // Current K tile and output window.
  val kernelRow = RegInit(0.U(4.W))
  val kernelCol = RegInit(0.U(4.W))
  val inGroup   = RegInit(0.U(groupWidth.W))
  val outGroup  = RegInit(0.U(groupWidth.W))
  val localTile = RegInit(0.U(16.W))
  val window    = RegInit(0.U(11.W))
  val windowRow = RegInit(0.U(16.W))
  val windowCol = RegInit(0.U(16.W))
  val rowData   = Reg(UInt(128.W))

  val div = Instantiate(new RestoringDiv(16, 8, ballParam))
  div.io.start := false.B
  div.io.a     := 0.U
  div.io.b     := 0.U

  val padded        = size +& (padding << 1)
  val tileRows      = 1.U(5.W) << tileShift
  val paddedWindows = ((windowCount +& tileRows - 1.U) >> tileShift) << tileShift
  val sourceRow     = startRow +& windowRow * stride +& kernelRow
  val sourceCol     = startCol +& windowCol * stride +& kernelCol
  val inBounds      = sourceRow >= padding && sourceRow < padding +& size &&
    sourceCol >= padding && sourceCol < padding +& size
  val sourceLine    = inputBase +& (sourceRow - padding) * size +& (sourceCol - padding)
  val outputLine    = (((window >> tileShift) * groupTiles +& localTile) << tileShift) |
    (window & (tileRows - 1.U))
  val lastTile      = kernelRow +& 1.U === kernel && kernelCol +& 1.U === kernel && inGroup +& 1.U === inGroups

  io.cmdReq.ready        := state === idle
  io.cmdResp.valid       := state === complete
  io.cmdResp.bits.rob_id := robId
  io.status.idle         := state === idle
  io.status.running      := state =/= idle && state =/= complete

  io.bankRead(0).rob_id           := robId
  io.bankRead(0).bank_id          := inBank
  io.bankRead(0).group_id         := inBase + inGroup
  io.bankRead(0).io.req.valid     := state === readReq
  io.bankRead(0).io.req.bits.addr := sourceLine(addressWidth - 1, 0)
  io.bankRead(0).io.resp.ready    := state === readResp

  io.bankWrite(0).rob_id           := robId
  io.bankWrite(0).bank_id          := outBank
  io.bankWrite(0).group_id         := outBase + outGroup
  io.bankWrite(0).io.req.valid     := state === writeReq
  io.bankWrite(0).io.req.bits.addr := outputLine(addressWidth - 1, 0)
  io.bankWrite(0).io.req.bits.data := rowData
  io.bankWrite(0).io.req.bits.mask := VecInit(Seq.fill(memParam.bankMaskLen)(true.B))
  io.bankWrite(0).io.resp.ready    := state === writeResp

  switch(state) {
    is(idle) {
      when(io.cmdReq.fire) {
        val cmd       = io.cmdReq.bits
        val tileValue = cmd.rs2(50, 46)
        assert(cmd.rs2(63, 51) === 0.U, "IM2COL reserves rs2[63:51]")
        assert(
          cmd.iter =/= 0.U && cmd.iter <= maxIter.U,
          "IM2COL input size must be in the configured range"
        )
        assert(
          cmd.rs2(3, 0) =/= 0.U && cmd.rs2(3, 0) <= maxKSize.U,
          "IM2COL kernel must be in the configured range"
        )
        assert(cmd.rs2(7, 4) =/= 0.U, "IM2COL stride must be positive")
        assert(cmd.rs2(11, 8) <= maxPad.U, "IM2COL padding exceeds the configured range")
        assert(
          cmd.rs2(15, 12) <= cmd.rs2(11, 8) && cmd.rs2(19, 16) <= cmd.rs2(11, 8),
          "IM2COL start must not exceed padding"
        )
        assert(cmd.rs2(45, 36) =/= 0.U, "IM2COL window count must be positive")
        assert(PopCount(tileValue) === 1.U && tileValue <= 16.U, "IM2COL tile rows must be 1, 2, 4, 8, or 16")
        assert(
          cmd.rs2(25, 20) +& cmd.iter * cmd.iter <= memParam.bankEntries.U,
          "IM2COL input footprint exceeds bank depth"
        )
        assert(
          cmd.op1_col =/= 0.U && cmd.op2_col === 0.U && cmd.wr_col =/= 0.U &&
            cmd.op1_col <= memParam.bankNum.U && cmd.wr_col <= memParam.bankNum.U,
          "IM2COL requires input and output groups"
        )
        assert(cmd.op1_bank =/= cmd.wr_bank, "IM2COL input and output banks must differ")
        assert(
          cmd.iter +& (cmd.rs2(11, 8) << 1) >= cmd.rs2(3, 0) +& cmd.rs2(19, 16) &&
            cmd.iter +& (cmd.rs2(11, 8) << 1) >= cmd.rs2(3, 0) +& cmd.rs2(15, 12),
          "IM2COL kernel and start exceed padded input"
        )

        robId        := cmd.rob_id
        inBank       := cmd.op1_bank
        outBank      := cmd.wr_bank
        inBase       := cmd.op1_group
        outBase      := cmd.wr_group
        inGroups     := cmd.op1_col
        outGroups    := cmd.wr_col
        size         := cmd.iter
        kernel       := cmd.rs2(3, 0)
        stride       := cmd.rs2(7, 4)
        padding      := cmd.rs2(11, 8)
        startCol     := cmd.rs2(15, 12)
        startRow     := cmd.rs2(19, 16)
        inputBase    := cmd.rs2(25, 20)
        windowStart  := cmd.rs2(35, 26)
        windowCount  := cmd.rs2(45, 36)
        tileShift    := OHToUInt(tileValue)
        state        := divRows
        div.io.start := true.B
        div.io.a     := cmd.iter +& (cmd.rs2(11, 8) << 1) - cmd.rs2(3, 0) - cmd.rs2(19, 16)
        div.io.b     := cmd.rs2(7, 4)
      }
    }

    is(divRows) {
      when(div.io.done) {
        assert(div.io.r === 0.U, "IM2COL output row shape must be integral")
        outRows      := div.io.q + 1.U
        div.io.start := true.B
        div.io.a     := padded - kernel - startCol
        div.io.b     := stride
        state        := divCols
      }
    }

    is(divCols) {
      when(div.io.done) {
        assert(div.io.r === 0.U, "IM2COL output column shape must be integral")
        outCols      := div.io.q + 1.U
        div.io.start := true.B
        div.io.a     := windowStart
        div.io.b     := div.io.q + 1.U
        state        := divStart
      }
    }

    is(divStart) {
      when(div.io.done) {
        firstRow     := div.io.q
        firstCol     := div.io.r
        div.io.start := true.B
        div.io.a     := kernel * kernel * inGroups
        div.io.b     := outGroups
        state        := divTiles
      }
    }

    is(divTiles) {
      when(div.io.done) {
        assert(div.io.r === 0.U, "IM2COL output groups must divide the K tiles")
        groupTiles := div.io.q
        state      := check
      }
    }

    is(check) {
      assert(
        windowStart +& windowCount <= outRows * outCols,
        "IM2COL selected window range exceeds the output shape"
      )
      assert(
        paddedWindows * groupTiles <= memParam.bankEntries.U,
        "IM2COL output footprint exceeds bank depth"
      )
      kernelRow := 0.U
      kernelCol := 0.U
      inGroup   := 0.U
      outGroup  := 0.U
      localTile := 0.U
      window    := 0.U
      windowRow := firstRow
      windowCol := firstCol
      state     := select
    }

    // The current output row is an input row, or zero for padding and for
    // windows past windowCount.
    is(select) {
      when(window < windowCount && inBounds) {
        state := readReq
      }.otherwise {
        rowData := 0.U
        state   := writeReq
      }
    }

    is(readReq) {
      when(io.bankRead(0).io.req.fire) {
        state := readResp
      }
    }

    is(readResp) {
      when(io.bankRead(0).io.resp.fire) {
        rowData := io.bankRead(0).io.resp.bits.data
        state   := writeReq
      }
    }

    is(writeReq) {
      when(io.bankWrite(0).io.req.fire) {
        state := writeResp
      }
    }

    is(writeResp) {
      when(io.bankWrite(0).io.resp.fire) {
        when(window +& 1.U =/= paddedWindows) {
          window := window + 1.U
          when(windowCol +& 1.U === outCols) {
            windowCol := 0.U
            windowRow := windowRow + 1.U
          }.otherwise {
            windowCol := windowCol + 1.U
          }
          state  := select
        }.elsewhen(lastTile) {
          state := complete
        }.otherwise {
          window    := 0.U
          windowRow := firstRow
          windowCol := firstCol
          when(inGroup +& 1.U === inGroups) {
            inGroup := 0.U
            when(kernelCol +& 1.U === kernel) {
              kernelCol := 0.U
              kernelRow := kernelRow + 1.U
            }.otherwise {
              kernelCol := kernelCol + 1.U
            }
          }.otherwise {
            inGroup := inGroup + 1.U
          }
          when(localTile +& 1.U === groupTiles) {
            localTile := 0.U
            outGroup  := outGroup + 1.U
          }.otherwise {
            localTile := localTile + 1.U
          }
          state     := select
        }
      }
    }

    is(complete) {
      when(io.cmdResp.fire) {
        state := idle
      }
    }
  }
}
