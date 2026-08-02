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
    .getOrElse(throw new IllegalArgumentException("TransposeBall not found in config"))

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
  val dstIdx   = RegInit(0.U(32.W))
  val pending  = RegInit(false.B)
  val wrData   = RegInit(0.U(bankWidth.W))
  val wrMask   = RegInit(VecInit(Seq.fill(b.memDomain.bankMaskLen)(0.U(1.W))))
  val wrGroup  = RegInit(0.U(log2Up(b.memDomain.bankNum + 1).W))
  val wrAddr   = RegInit(0.U(b.frontend.iter_len.W))
  val wrValid  = RegInit(false.B)
  val beatFill = RegInit(0.U(8.W))

  val cached     = RegInit(false.B)
  val cacheAddr  = RegInit(0.U(b.frontend.iter_len.W))
  val cacheGroup = RegInit(0.U(log2Up(b.memDomain.bankNum + 1).W))
  val cacheData  = RegInit(0.U(bankWidth.W))

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
    io.bankWrite(i).io.req.bits.mask := VecInit(Seq.fill(b.memDomain.bankMaskLen)(0.U(1.W)))
    io.bankWrite(i).io.resp.ready    := (state =/= idle)
  }

  io.cmdReq.ready            := (state === idle)
  io.cmdResp.valid           := false.B
  io.cmdResp.bits.rob_id     := rob_id_reg
  io.cmdResp.bits.is_sub     := is_sub_reg
  io.cmdResp.bits.sub_rob_id := sub_rob_id_reg

  val elemBits = elem_reg
  val elemBytes = elemBits >> 3
  val epg       = (rowBytes.U / elemBytes)
  val wElems    = ncol_reg * epg
  val total     = iter_reg * wElems

  // Decode dest linear index -> (virt_row, group, lane) and src (r,c).
  val dCol = Mux(iter_reg === 0.U, 0.U, dstIdx / iter_reg) // c in W×iter
  val dRow = Mux(iter_reg === 0.U, 0.U, dstIdx % iter_reg) // r
  val srcR = dRow
  val srcC = dCol

  val srcGroup = srcC / epg
  val srcLane  = srcC % epg
  val srcAddr  = srcR

  val dstVirtRow = Mux(wElems === 0.U, 0.U, dstIdx / wElems)
  val dstVirtCol = Mux(wElems === 0.U, 0.U, dstIdx % wElems)
  val dstGroup   = dstVirtCol / epg
  val dstLane    = dstVirtCol % epg

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
        dstIdx         := 0.U
        pending        := false.B
        wrValid        := false.B
        beatFill       := 0.U
        cached         := false.B
        assert(cmd.iter > 0.U, "Transpose iter must be > 0")
        assert(cmd.op1_bank =/= cmd.wr_bank, "Transpose op1 and wr must differ")
        assert(cmd.op1_col === cmd.wr_col && cmd.op1_col =/= 0.U, "Transpose cols mismatch")
        assert(cmd.rs2(63, 8) === 0.U, "Transpose rs2[63:8] must be 0")
        assert(cmd.rs2(7, 0) === 8.U || cmd.rs2(7, 0) === 32.U, "Transpose elem_bits")
        assert(bankWidth.U % cmd.rs2(7, 0) === 0.U,
          "Transpose bankWidth not divisible by elem_bits")
        state := sRead
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
        val shift = srcLane * elemBytes * 8.U
        val mask  = (1.U << (elemBytes * 8.U)) - 1.U
        val elem  = (cacheData >> shift) & mask
        val wsh   = dstLane * elemBytes * 8.U
        val base  = Mux(beatFill === 0.U, 0.U, wrData)
        when(beatFill === 0.U) {
          wrGroup := dstGroup
          wrAddr  := dstVirtRow
        }
        wrData   := base | (elem << wsh)
        beatFill := beatFill + 1.U
        dstIdx   := dstIdx + 1.U

        val beatDone = (beatFill + 1.U === epg) || (dstIdx + 1.U === total)
        when(beatDone) {
          wrMask.foreach(_ := 1.U)
          wrValid := true.B
          state   := sWrite
        }
      }
    }

    is(sWrite) {
      io.bankWrite(0).group_id         := wrGroup
      io.bankWrite(0).io.req.valid     := wrValid
      io.bankWrite(0).io.req.bits.addr := wrAddr
      io.bankWrite(0).io.req.bits.data := wrData
      io.bankWrite(0).io.req.bits.mask := wrMask
      when(io.bankWrite(0).io.req.fire) {
        wrValid  := false.B
        beatFill := 0.U
        when(dstIdx === total) {
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
