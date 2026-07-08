package examples.balls.systolicarray

import chisel3._
import chisel3.util._
import chisel3.stage._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.memdomain.backend.banks.{SramReadReq, SramReadResp}
import framework.top.GlobalConfig
import examples.balls.systolicarray.configs.SystolicBallParam

@instantiable
class SystolicArrayLoad(val b: GlobalConfig) extends Module {
  val config       = SystolicBallParam()
  val InputNum     = config.lane
  val bankWidth    = b.memDomain.bankWidth
  val inputWidth   = config.inputWidth
  val elemsPerLine = bankWidth / inputWidth
  val readsPerRow  = if (InputNum <= elemsPerLine) 1 else InputNum / elemsPerLine
  val ballMapping  = b.ballDomain.ballIdMappings.find(_.ballName == "SystolicArrayBall")
    .getOrElse(throw new IllegalArgumentException("SystolicArrayBall not found in config"))
  val inBW         = ballMapping.inBW

  require(InputNum == config.InputNum)
  require(InputNum <= elemsPerLine || InputNum % elemsPerLine == 0)
  require(readsPerRow == 1 || inBW >= 2)

  @public
  val io = IO(new Bundle {
    val bankReadReq  = Vec(inBW, Decoupled(new SramReadReq(b)))
    val bankReadResp = Vec(inBW, Flipped(Decoupled(new SramReadResp(b))))
    val ctrl_ld_i    = Flipped(Decoupled(new ctrl_ld_req(b)))
    val ld_ex_o      = Decoupled(new ld_ex_req(b))
    val op1_bank_o   = Output(UInt(log2Up(b.memDomain.bankNum).W))
    val op2_bank_o   = Output(UInt(log2Up(b.memDomain.bankNum).W))
  })

  val op1_bank            = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  val op2_bank            = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  val op1_addr            = RegInit(0.U(log2Up(b.memDomain.bankEntries).W))
  val op2_addr            = RegInit(0.U(log2Up(b.memDomain.bankEntries).W))
  val iter                = RegInit(0.U(b.frontend.iter_len.W))
  val op1_iter_counter    = RegInit(0.U(b.frontend.iter_len.W))
  val op2_iter_counter    = RegInit(0.U(b.frontend.iter_len.W))
  val idle :: busy :: Nil = Enum(2)
  val state               = RegInit(idle)
  val ld_ex_iter_reg      = RegInit(0.U(b.frontend.iter_len.W))

  val bankRespQueue0 = Module(new Queue(new SramReadResp(b), entries = 8))
  val bankRespQueue1 = Module(new Queue(new SramReadResp(b), entries = 8))

  for (i <- 0 until inBW) {
    io.bankReadReq(i).valid     := false.B
    io.bankReadReq(i).bits.addr := 0.U
  }

  io.op1_bank_o      := op1_bank
  io.op2_bank_o      := op2_bank
  io.ctrl_ld_i.ready := state === idle

  bankRespQueue0.io.enq <> io.bankReadResp(0)
  bankRespQueue1.io.enq <> io.bankReadResp(1)

  when(io.ctrl_ld_i.fire) {
    op1_bank         := io.ctrl_ld_i.bits.op1_bank
    op2_bank         := io.ctrl_ld_i.bits.op2_bank
    op1_addr         := io.ctrl_ld_i.bits.op1_bank_addr
    op2_addr         := io.ctrl_ld_i.bits.op2_bank_addr
    iter             := io.ctrl_ld_i.bits.iter
    op1_iter_counter := 0.U
    op2_iter_counter := 0.U
    state            := busy
  }

  if (inBW <= 2) {
    when(state === busy && io.ld_ex_o.ready) {
      io.bankReadReq(0).valid     := op1_iter_counter < iter
      io.bankReadReq(0).bits.addr := op1_addr + op1_iter_counter
      op1_iter_counter            := Mux(io.bankReadReq(0).ready, op1_iter_counter + 1.U, op1_iter_counter)
    }

    when(state === busy && io.ld_ex_o.ready) {
      io.bankReadReq(1).valid     := op2_iter_counter < iter
      io.bankReadReq(1).bits.addr := op2_addr + op2_iter_counter
      op2_iter_counter            := Mux(io.bankReadReq(1).ready, op2_iter_counter + 1.U, op2_iter_counter)
    }

    val both_valid = bankRespQueue0.io.deq.valid && bankRespQueue1.io.deq.valid

    io.ld_ex_o.valid := both_valid
    when(both_valid) {
      val op1Line = bankRespQueue0.io.deq.bits.data.asTypeOf(Vec(elemsPerLine, UInt(inputWidth.W)))
      val op2Line = bankRespQueue1.io.deq.bits.data.asTypeOf(Vec(elemsPerLine, UInt(inputWidth.W)))
      io.ld_ex_o.bits.op1  := VecInit(op1Line.slice(0, InputNum))
      io.ld_ex_o.bits.op2  := VecInit(op2Line.slice(0, InputNum))
      io.ld_ex_o.bits.iter := ld_ex_iter_reg
    }.otherwise {
      io.ld_ex_o.bits.iter := 0.U
      io.ld_ex_o.bits.op1  := VecInit(Seq.fill(InputNum)(0.U(inputWidth.W)))
      io.ld_ex_o.bits.op2  := VecInit(Seq.fill(InputNum)(0.U(inputWidth.W)))
    }

    bankRespQueue0.io.deq.ready := io.ld_ex_o.fire
    bankRespQueue1.io.deq.ready := io.ld_ex_o.fire
  } else if (inBW >= 4) {
    val bankRespQueue2 = Module(new Queue(new SramReadResp(b), entries = 8))
    val bankRespQueue3 = Module(new Queue(new SramReadResp(b), entries = 8))

    bankRespQueue2.io.enq <> io.bankReadResp(2)
    bankRespQueue3.io.enq <> io.bankReadResp(3)

    when(state === busy && ld_ex_iter_reg < iter && op1_iter_counter < (iter / inBW.U * 2.U)) {
      io.bankReadReq(0).valid     := true.B
      io.bankReadReq(0).bits.addr := op1_addr + op1_iter_counter
      io.bankReadReq(1).valid     := true.B
      io.bankReadReq(1).bits.addr := op1_addr + op1_iter_counter
      op1_iter_counter            := Mux(io.bankReadReq(0).ready, op1_iter_counter + 1.U, op1_iter_counter)
      io.bankReadReq(2).valid     := true.B
      io.bankReadReq(2).bits.addr := op2_addr + op2_iter_counter
      io.bankReadReq(3).valid     := true.B
      io.bankReadReq(3).bits.addr := op2_addr + op2_iter_counter
      op2_iter_counter            := Mux(io.bankReadReq(2).ready, op2_iter_counter + 1.U, op2_iter_counter)
    }

    val all_valid = bankRespQueue0.io.deq.valid && bankRespQueue1.io.deq.valid &&
      bankRespQueue2.io.deq.valid && bankRespQueue3.io.deq.valid

    io.ld_ex_o.valid := all_valid
    when(all_valid) {
      io.ld_ex_o.bits.op1  := Cat(
        bankRespQueue1.io.deq.bits.data,
        bankRespQueue0.io.deq.bits.data
      ).asTypeOf(Vec(InputNum, UInt(inputWidth.W)))
      io.ld_ex_o.bits.op2  := Cat(
        bankRespQueue3.io.deq.bits.data,
        bankRespQueue2.io.deq.bits.data
      ).asTypeOf(Vec(InputNum, UInt(inputWidth.W)))
      io.ld_ex_o.bits.iter := ld_ex_iter_reg
    }.otherwise {
      io.ld_ex_o.bits.iter := 0.U
      io.ld_ex_o.bits.op1  := VecInit(Seq.fill(InputNum)(0.U(inputWidth.W)))
      io.ld_ex_o.bits.op2  := VecInit(Seq.fill(InputNum)(0.U(inputWidth.W)))
    }

    bankRespQueue0.io.deq.ready := io.ld_ex_o.fire
    bankRespQueue1.io.deq.ready := io.ld_ex_o.fire
    bankRespQueue2.io.deq.ready := io.ld_ex_o.fire
    bankRespQueue3.io.deq.ready := io.ld_ex_o.fire
  } else {
    val op1Half   = Reg(UInt(bankWidth.W))
    val op2Half   = Reg(UInt(bankWidth.W))
    val readPhase = RegInit(false.B)

    val rowBase     = ld_ex_iter_reg * readsPerRow.U
    val queuesEmpty = !bankRespQueue0.io.deq.valid && !bankRespQueue1.io.deq.valid

    when(state === busy && ld_ex_iter_reg < iter && queuesEmpty) {
      io.bankReadReq(0).valid     := true.B
      io.bankReadReq(0).bits.addr := op1_addr + rowBase + readPhase
      io.bankReadReq(1).valid     := true.B
      io.bankReadReq(1).bits.addr := op2_addr + rowBase + readPhase
    }

    val both_valid = bankRespQueue0.io.deq.valid && bankRespQueue1.io.deq.valid

    when(both_valid && !readPhase) {
      op1Half   := bankRespQueue0.io.deq.bits.data
      op2Half   := bankRespQueue1.io.deq.bits.data
      readPhase := true.B
    }

    io.ld_ex_o.valid := both_valid && readPhase
    when(both_valid && readPhase) {
      io.ld_ex_o.bits.op1  := Cat(
        bankRespQueue0.io.deq.bits.data,
        op1Half
      ).asTypeOf(Vec(InputNum, UInt(inputWidth.W)))
      io.ld_ex_o.bits.op2  := Cat(
        bankRespQueue1.io.deq.bits.data,
        op2Half
      ).asTypeOf(Vec(InputNum, UInt(inputWidth.W)))
      io.ld_ex_o.bits.iter := ld_ex_iter_reg
    }.otherwise {
      io.ld_ex_o.bits.iter := 0.U
      io.ld_ex_o.bits.op1  := VecInit(Seq.fill(InputNum)(0.U(inputWidth.W)))
      io.ld_ex_o.bits.op2  := VecInit(Seq.fill(InputNum)(0.U(inputWidth.W)))
    }

    bankRespQueue0.io.deq.ready := both_valid && (!readPhase || io.ld_ex_o.fire)
    bankRespQueue1.io.deq.ready := both_valid && (!readPhase || io.ld_ex_o.fire)

    when(io.ctrl_ld_i.fire) {
      readPhase := false.B
    }

    when(io.ld_ex_o.fire) {
      readPhase := false.B
    }
  }

  when(io.ld_ex_o.fire) {
    ld_ex_iter_reg := ld_ex_iter_reg + 1.U
  }

  when(state === busy && ld_ex_iter_reg === iter) {
    state            := idle
    op1_iter_counter := 0.U
    op2_iter_counter := 0.U
    ld_ex_iter_reg   := 0.U
  }
}
