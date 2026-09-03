package examples.balls.transpose

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}

import framework.balldomain.rs.{BallRsComplete, BallRsIssue}
import framework.balldomain.blink.{BallStatus, BankRead, BankWrite}
import framework.top.GlobalConfig
import examples.balls.transpose.configs.TransposeBallParam

@instantiable
class Transpose(val b: GlobalConfig) extends Module {
  val ballConfig = TransposeBallParam(b)
  val bankWidth  = b.memDomain.bankWidth
  val rowBytes   = bankWidth / 8

  val ballMapping = b.ballDomain.ballIdMappings
    .find(_.ballName == "TransposeBall")
    .getOrElse(
      throw new IllegalArgumentException("TransposeBall not found in config")
    )

  val inBW  = ballMapping.inBW
  val outBW = ballMapping.outBW
  require(inBW == outBW, "TransposeBall requires inBW == outBW")
  require(inBW == 1, "TransposeBall gather/scatter path requires inBW == 1")
  require(bankWidth % 8 == 0, "bankWidth must be byte-aligned")
  require(
    ballConfig.InputNum * ballConfig.inputWidth == bankWidth,
    "TransposeBall InputNum*inputWidth must equal bankWidth"
  )

  @public
  val io = IO(new Bundle {
    val cmdReq    = Flipped(Decoupled(new BallRsIssue(b)))
    val cmdResp   = Decoupled(new BallRsComplete(b))
    val bankRead  = Vec(inBW, Flipped(new BankRead(b)))
    val bankWrite = Vec(outBW, Flipped(new BankWrite(b)))
    val status    = new BallStatus
  })

  val rob_id_reg     = RegInit(0.U(log2Up(b.frontend.rob_entries).W))
  val is_sub_reg     = RegInit(false.B)
  val sub_rob_id_reg = RegInit(0.U(log2Up(b.frontend.sub_rob_depth * 4).W))

  val rbank_reg = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  val wbank_reg = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  val ncol_reg  = RegInit(0.U(log2Up(b.memDomain.bankNum + 1).W))
  val iter_reg  = RegInit(0.U(b.frontend.iter_len.W))
  val elem_reg  = RegInit(0.U(8.W))

  val idle :: sRead :: sWrite :: complete :: Nil = Enum(4)
  val state                                      = RegInit(idle)

  // Walk destination dense index 0 .. iter*W-1, filling write beats.
  val pending  = RegInit(false.B)
  val wrBytes  = Reg(Vec(rowBytes, UInt(8.W)))
  val wrMask   = RegInit(VecInit(Seq.fill(b.memDomain.bankMaskLen)(0.U(1.W))))
  val wrGroup  = RegInit(0.U(log2Up(b.memDomain.bankNum + 1).W))
  val wrAddr   = RegInit(0.U(b.frontend.iter_len.W))
  val wrValid  = RegInit(false.B)
  val beatFill = RegInit(0.U(8.W))

  val cached     = RegInit(false.B)
  val cacheAddr  = RegInit(0.U(b.frontend.iter_len.W))
  val cacheGroup = RegInit(0.U(log2Up(b.memDomain.bankNum + 1).W))
  val cacheData  = RegInit(0.U(bankWidth.W))

  val srcR         = RegInit(0.U(b.frontend.iter_len.W))
  val srcC         = RegInit(0.U(32.W))
  val dstVirtRow   = RegInit(0.U(b.frontend.iter_len.W))
  val dstVirtCol   = RegInit(0.U(32.W))
  val epgReg       = RegInit(0.U(8.W))
  val wElemsReg    = RegInit(0.U(32.W))
  val elemBytesReg = RegInit(0.U(8.W))
  val lastBeat     = RegInit(false.B)

  for (i <- 0 until inBW) {
    io.bankRead(i).rob_id           := rob_id_reg
    io.bankRead(i).ball_id          := 0.U
    io.bankRead(i).bank_id          := rbank_reg
    io.bankRead(i).group_id         := 0.U
    io.bankRead(i).io.req.valid     := false.B
    io.bankRead(i).io.req.bits.addr := 0.U
    io.bankRead(i).io.resp.ready    := false.B
  }
  for (i <- 0 until outBW) {
    io.bankWrite(i).rob_id           := rob_id_reg
    io.bankWrite(i).ball_id          := 0.U
    io.bankWrite(i).bank_id          := wbank_reg
    io.bankWrite(i).group_id         := 0.U
    io.bankWrite(i).io.req.valid     := false.B
    io.bankWrite(i).io.req.bits.addr := 0.U
    io.bankWrite(i).io.req.bits.data := 0.U
    io.bankWrite(i).io.req.bits.mask := VecInit(
      Seq.fill(b.memDomain.bankMaskLen)(0.U(1.W))
    )
    io.bankWrite(i).io.resp.ready    := (state =/= idle)
  }

  io.cmdReq.ready            := (state === idle)
  io.cmdResp.valid           := false.B
  io.cmdResp.bits.rob_id     := rob_id_reg
  io.cmdResp.bits.is_sub     := is_sub_reg
  io.cmdResp.bits.sub_rob_id := sub_rob_id_reg

  require(isPow2(rowBytes), "Transpose rowBytes must be power of 2")

  val laneBitsI8  = log2Ceil(rowBytes)
  val laneBitsI32 = log2Ceil(rowBytes / 4)

  val isI8     = elemBytesReg === 1.U
  val srcLane  = Mux(isI8, srcC(laneBitsI8 - 1, 0), srcC(laneBitsI32 - 1, 0))
  val srcGroup = Mux(isI8, srcC >> laneBitsI8.U, srcC >> laneBitsI32.U)
    .asTypeOf(UInt(log2Up(b.memDomain.bankNum + 1).W))
  val srcAddr  = srcR

  val dstGroup = Mux(isI8, dstVirtCol >> laneBitsI8.U, dstVirtCol >> laneBitsI32.U)
    .asTypeOf(UInt(log2Up(b.memDomain.bankNum + 1).W))
  val bytes    = cacheData.asTypeOf(Vec(rowBytes, UInt(8.W)))
  val words    = cacheData.asTypeOf(Vec(rowBytes / 4, UInt(32.W)))

  val needRead = !cached || cacheAddr =/= srcAddr || cacheGroup =/= srcGroup

  switch(state) {
    is(idle) {
      when(io.cmdReq.fire) {
        val cmd = io.cmdReq.bits.cmd
        rob_id_reg     := io.cmdReq.bits.rob_id
        is_sub_reg     := io.cmdReq.bits.is_sub
        sub_rob_id_reg := io.cmdReq.bits.sub_rob_id
        rbank_reg      := cmd.op1_bank
        wbank_reg      := cmd.wr_bank
        ncol_reg       := cmd.op1_col
        iter_reg       := cmd.iter
        elem_reg       := cmd.rs2(7, 0)
        pending        := false.B
        wrValid        := false.B
        beatFill       := 0.U
        cached         := false.B
        srcR           := 0.U
        srcC           := 0.U
        dstVirtRow     := 0.U
        dstVirtCol     := 0.U
        lastBeat       := false.B
        val i8 = cmd.rs2(7, 0) === 8.U
        elemBytesReg := Mux(i8, 1.U, 4.U)
        epgReg       := Mux(i8, rowBytes.U, (rowBytes / 4).U)
        wElemsReg    := Mux(i8, cmd.op1_col << laneBitsI8.U, cmd.op1_col << laneBitsI32.U)
        assert(cmd.iter > 0.U, "Transpose iter must be > 0")
        assert(cmd.op1_bank =/= cmd.wr_bank, "Transpose op1 and wr must differ")
        assert(
          cmd.op1_col === cmd.wr_col && cmd.op1_col =/= 0.U,
          "Transpose cols mismatch"
        )
        assert(cmd.rs2(63, 8) === 0.U, "Transpose rs2[63:8] must be 0")
        assert(
          cmd.rs2(7, 0) === 8.U || cmd.rs2(7, 0) === 32.U,
          "Transpose elem_bits"
        )
        assert(
          bankWidth.U % cmd.rs2(7, 0) === 0.U,
          "Transpose bankWidth not divisible by elem_bits"
        )
        state        := sRead
      }
    }

    is(sRead) {
      io.bankRead(0).group_id         := srcGroup
      io.bankRead(0).io.resp.ready    := pending
      io.bankRead(0).io.req.valid     := needRead && !pending && !wrValid
      io.bankRead(0).io.req.bits.addr := srcAddr

      when(io.bankRead(0).io.req.fire) {
        pending := true.B
      }
      when(io.bankRead(0).io.resp.fire) {
        cacheData  := io.bankRead(0).io.resp.bits.data
        cacheAddr  := srcAddr
        cacheGroup := srcGroup
        cached     := true.B
        pending    := false.B
      }

      when(cached && !needRead && !wrValid) {
        when(beatFill === 0.U) {
          wrGroup := dstGroup
          wrAddr  := dstVirtRow
          wrBytes := VecInit(Seq.fill(rowBytes)(0.U(8.W)))
        }
        when(isI8) {
          wrBytes(beatFill) := bytes(srcLane)
        }.otherwise {
          val w = words(srcLane)
          wrBytes(beatFill << 2)         := w(7, 0)
          wrBytes((beatFill << 2) + 1.U) := w(15, 8)
          wrBytes((beatFill << 2) + 2.U) := w(23, 16)
          wrBytes((beatFill << 2) + 3.U) := w(31, 24)
        }
        beatFill := beatFill + 1.U

        val lastElem = (srcR === iter_reg - 1.U) && (srcC === wElemsReg - 1.U)
        when(srcR === iter_reg - 1.U) {
          srcR := 0.U
          srcC := srcC + 1.U
        }.otherwise {
          srcR := srcR + 1.U
        }
        when(dstVirtCol === wElemsReg - 1.U) {
          dstVirtCol := 0.U
          dstVirtRow := dstVirtRow + 1.U
        }.otherwise {
          dstVirtCol := dstVirtCol + 1.U
        }

        val beatDone = (beatFill + 1.U === epgReg) || lastElem
        when(beatDone) {
          wrMask.foreach(_ := 1.U)
          wrValid          := true.B
          lastBeat         := lastElem
          state            := sWrite
        }
      }
    }

    is(sWrite) {
      io.bankWrite(0).group_id         := wrGroup
      io.bankWrite(0).io.req.valid     := wrValid
      io.bankWrite(0).io.req.bits.addr := wrAddr
      io.bankWrite(0).io.req.bits.data := wrBytes.asUInt
      io.bankWrite(0).io.req.bits.mask := wrMask
      io.bankWrite(0).io.resp.ready    := true.B
      when(io.bankWrite(0).io.req.fire) {
        wrValid  := false.B
        beatFill := 0.U
      }
      when(io.bankWrite(0).io.resp.fire) {
        when(lastBeat) {
          state := complete
        }.otherwise {
          state := sRead
        }
      }
    }

    is(complete) {
      io.cmdResp.valid := true.B
      when(io.cmdResp.fire) {
        state := idle
      }
    }
  }

  io.status.idle    := (state === idle)
  io.status.running := (state =/= idle)
}
