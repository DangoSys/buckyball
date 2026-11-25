package examples.balls.im2col

import framework.generic.contract.blink.BlinkParam

import framework.generic.contract.blink.BallResponse

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import framework.ball.blink.BallStatus
import framework.generic.queue.Queue
import examples.balls.im2col.configs.Im2colBallParam
import framework.generic.contract.configs.BallRouteParam
import framework.generic.contract.blink.BallCommand
import framework.generic.contract.memdomain.BankRead
import framework.generic.contract.memdomain.BankWrite

class Im2colRow(addressWidth: Int, groupWidth: Int) extends Bundle {
  val data    = UInt(128.W)
  val address = UInt(addressWidth.W)
  val group   = UInt(groupWidth.W)
  val last    = Bool()
}

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
  link:          BlinkParam,
  instance:      BallRouteParam,
  val ballParam: Im2colBallParam)
    extends Module {
  val maxIter      = ballParam.maxIter
  val maxKSize     = ballParam.maxKSize
  val maxPad       = ballParam.maxPadding
  val addressWidth = log2Ceil(link.bankEntries)
  val groupWidth   = log2Up(link.bankNum + 1)

  require(instance.inBW == 1, "Im2colBall requires inBW=1")
  require(instance.outBW == 1, "Im2colBall requires outBW=1")
  require(link.bankWidth == 128, "Im2colBall requires 128-bit bank rows")
  require(ballParam.inputWidth == 8, "Im2colBall moves 16 INT8 channels per row")
  require(isPow2(link.bankEntries), "Im2colBall requires power-of-two bank depth")
  require(maxIter + 2 * maxPad <= 255, "Im2colBall dimensions must fit the divider")
  require(maxKSize <= 15 && maxPad <= 15, "Im2colBall kernel and padding fields are four bits")

  @public
  val io = IO(new Bundle {
    val cmdReq    = Flipped(Decoupled(new BallCommand(link)))
    val cmdResp   = Decoupled(new BallResponse(link))
    val bankRead  = Vec(1, Flipped(new BankRead(link)))
    val bankWrite = Vec(1, Flipped(new BankWrite(link)))
    val status    = new BallStatus
  })

  val idle :: divRows :: divCols :: divStart :: divTiles :: check :: select :: readResp :: drain :: complete :: Nil =
    Enum(10)
  val state                                                                                                         = RegInit(idle)

  val robId       = RegInit(0.U(link.robIdBits.W))
  val inBank      = RegInit(0.U(link.bankIdBits.W))
  val outBank     = RegInit(0.U(link.bankIdBits.W))
  val inBase      = RegInit(0.U(link.groupIdBits.W))
  val outBase     = RegInit(0.U(link.groupIdBits.W))
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

  val pendingRows = Instantiate(new Queue(
    new Im2colRow(addressWidth, groupWidth),
    2,
    false,
    false
  ))

  val writePending = RegInit(false.B)

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
  io.bankRead(0).io.req.valid     := state === select && window < windowCount && inBounds
  io.bankRead(0).io.req.bits.addr := sourceLine(addressWidth - 1, 0)
  io.bankRead(0).io.resp.ready    := state === readResp && pendingRows.io.enq.ready

  pendingRows.io.enq.valid        := (state === select && !(window < windowCount && inBounds)) ||
    (state === readResp && io.bankRead(0).io.resp.valid)
  pendingRows.io.enq.bits.data    := Mux(state === readResp, io.bankRead(0).io.resp.bits.data, 0.U)
  pendingRows.io.enq.bits.address := outputLine(addressWidth - 1, 0)
  pendingRows.io.enq.bits.group   := outGroup
  pendingRows.io.enq.bits.last    := lastTile && window +& 1.U === paddedWindows

  io.bankWrite(0).rob_id                         := robId
  io.bankWrite(0).bank_id                        := outBank
  io.bankWrite(0).group_id                       := outBase + pendingRows.io.deq.bits.group
  io.bankWrite(0).io.req.valid                   := pendingRows.io.deq.valid && !writePending
  io.bankWrite(0).io.req.bits.addr               := pendingRows.io.deq.bits.address
  io.bankWrite(0).io.req.bits.data               := pendingRows.io.deq.bits.data
  io.bankWrite(0).io.req.bits.mask               := VecInit(Seq.fill(link.bankMaskLen)(true.B))
  io.bankWrite(0).io.resp.ready                  := writePending
  pendingRows.io.deq.ready                       := io.bankWrite(0).io.resp.fire
  when(io.bankWrite(0).io.req.fire)(writePending := true.B)
  when(io.bankWrite(0).io.resp.fire) {
    writePending                             := false.B
    when(pendingRows.io.deq.bits.last)(state := complete)
  }

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
          cmd.rs2(25, 20) +& cmd.iter * cmd.iter <= link.bankEntries.U,
          "IM2COL input footprint exceeds bank depth"
        )
        assert(
          cmd.op1_col =/= 0.U && cmd.op2_col === 0.U && cmd.wr_col =/= 0.U &&
            cmd.op1_col <= link.bankNum.U && cmd.wr_col <= link.bankNum.U,
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
        paddedWindows * groupTiles <= link.bankEntries.U,
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
      when(io.bankRead(0).io.req.fire) {
        state := readResp
      }
    }
    is(complete) {
      when(io.cmdResp.fire) {
        state := idle
      }
    }
  }

  // Reading the next row overlaps the previous acknowledged write.
  when(pendingRows.io.enq.fire) {
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
      state := drain
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
