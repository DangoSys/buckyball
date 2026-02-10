package framework.balldomain.prototype.transpose

import chisel3._
import chisel3.util._
import chisel3.stage._
import chisel3.experimental.hierarchy.{instantiable, public}

import framework.balldomain.prototype.vector._
import framework.balldomain.rs.{BallRsComplete, BallRsIssue}
import framework.balldomain.blink.{BallStatus, BankRead, BankWrite}
import framework.top.GlobalConfig
import framework.balldomain.prototype.transpose.configs.TransposeBallParam

@instantiable
class Transpose(val b: GlobalConfig) extends Module {
  val ballConfig = TransposeBallParam()
  val InputNum   = ballConfig.InputNum
  val inputWidth = ballConfig.inputWidth
  val bankWidth  = b.memDomain.bankWidth

  // Get bandwidth from config
  val ballMapping = b.ballDomain.ballIdMappings.find(_.ballName == "TransposeBall")
    .getOrElse(throw new IllegalArgumentException("TransposeBall not found in config"))
  val inBW        = ballMapping.inBW
  val outBW       = ballMapping.outBW

  @public
  val io = IO(new Bundle {
    val cmdReq    = Flipped(Decoupled(new BallRsIssue(b)))
    val cmdResp   = Decoupled(new BallRsComplete(b))
    val bankRead  = Vec(inBW, Flipped(new BankRead(b)))
    val bankWrite = Vec(outBW, Flipped(new BankWrite(b)))
    val status    = new BallStatus
  })

  val rob_id_reg = RegInit(0.U(log2Up(b.frontend.rob_entries).W))
  when(io.cmdReq.fire) {
    rob_id_reg := io.cmdReq.bits.rob_id
  }

  for (i <- 0 until inBW) {
    io.bankRead(i).rob_id  := rob_id_reg
    io.bankRead(i).ball_id := 0.U
  }
  for (i <- 0 until outBW) {
    io.bankWrite(i).rob_id  := rob_id_reg
    io.bankWrite(i).ball_id := 0.U
  }

  val idle :: compute :: Nil = Enum(2)
  val state                  = RegInit(idle)

  // Matrix storage register (veclane x veclane)
  val regArray = Reg(Vec(2 * InputNum, Vec(InputNum, UInt(inputWidth.W))))

  // Counters
  val readCounter     = RegInit(0.U(10.W))
  val respCounter     = RegInit(0.U(10.W))
  val writeCounter    = RegInit(0.U(10.W))
  val respWaitcounter = RegInit(0.U(10.W))
  val writeHeadptr    = RegInit(0.U(10.W))
  val writeTailptr    = RegInit(0.U(10.W))

  // Instruction registers
  val robid_reg      = RegInit(0.U(10.W))
  val waddr_reg      = RegInit(0.U(10.W))
  val wbank_reg      = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  val raddr_reg      = RegInit(0.U(10.W))
  val rbank_reg      = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  val iter_reg       = RegInit(0.U(10.W))
  val write_iter_reg = RegInit(0.U(10.W))
  val mode_reg       = RegInit(0.U(1.W))

  // Precompute write data
  val writeDataReg = Reg(UInt(bankWidth.W))
  val writeMaskReg = Reg(Vec(b.memDomain.bankMaskLen, UInt(1.W)))

  val start_write = RegInit(false.B)
  // SRAM default assignment
  for (i <- 0 until inBW) {
    io.bankRead(i).io.req.valid     := false.B
    io.bankRead(i).io.req.bits.addr := 0.U
    io.bankRead(i).io.resp.ready    := false.B
  }

  for (i <- 0 until outBW) {
    io.bankWrite(i).io.req.valid      := false.B
    io.bankWrite(i).io.req.bits.addr  := 0.U
    io.bankWrite(i).io.req.bits.data  := 0.U
    io.bankWrite(i).io.req.bits.mask  := VecInit(Seq.fill(b.memDomain.bankMaskLen)(0.U(1.W)))
    io.bankWrite(i).io.req.bits.wmode := false.B
    io.bankWrite(i).io.resp.ready     := false.B
  }

  for (i <- 0 until inBW) {
    io.bankRead(i).bank_id := rbank_reg
  }
  for (i <- 0 until outBW) {
    io.bankWrite(i).bank_id := wbank_reg
  }

  // cmd interface default assignment
  io.cmdReq.ready        := state === idle
  io.cmdResp.valid       := false.B
  io.cmdResp.bits.rob_id := rob_id_reg

  when(state === idle && io.cmdReq.fire) {
    state           := compute
    readCounter     := 0.U
    respCounter     := 0.U
    respWaitcounter := io.cmdReq.bits.cmd.iter + respWaitcounter

    robid_reg := io.cmdReq.bits.rob_id
    waddr_reg := 0.U
    wbank_reg := io.cmdReq.bits.cmd.wr_bank
    raddr_reg := 0.U
    rbank_reg := io.cmdReq.bits.cmd.op1_bank
    iter_reg  := io.cmdReq.bits.cmd.iter
    mode_reg  := io.cmdReq.bits.cmd.special(0)
  }
  // read req
  when(((mode_reg === 1.U) && (state === compute) && RegNext(io.bankWrite(0).io.req.ready)) ||
    ((mode_reg === 0.U) && (state === compute))) {
    readCounter                     := readCounter + 1.U
    io.bankRead(0).io.req.valid     := readCounter < iter_reg
    io.bankRead(0).io.req.bits.addr := raddr_reg + readCounter
    state                           := Mux((readCounter >= iter_reg - 1.U) && io.cmdResp.ready, idle, state)
  }
  io.cmdResp.valid       := (readCounter >= (iter_reg - 1.U)) && (state === compute)
  io.cmdResp.bits.rob_id := robid_reg

  // read resp
  io.bankRead(0).io.resp.ready := true.B
  val dataWord = io.bankRead(0).io.resp.bits.data
  val row      = respCounter(4, 0)
  when(io.bankRead(0).io.resp.fire && respWaitcounter > 0.U) {
    for (col <- 0 until InputNum) {
      val hi = (col + 1) * inputWidth - 1
      val lo = col * inputWidth
      regArray(row)(col) := dataWord(hi, lo)
    }
    respCounter := Mux(respCounter === iter_reg - 1.U, 0.U, respCounter + 1.U)
    writeHeadptr    := Mux(writeHeadptr === 2.U * InputNum.U - 1.U, 0.U, writeHeadptr + 1.U)
    respWaitcounter := Mux(
      state === idle && io.cmdReq.fire,
      io.cmdReq.bits.cmd.iter + respWaitcounter - 1.U,
      respWaitcounter - 1.U
    )
  }

  // write req
  val wreg       = RegInit(0.U(10.W))
  val array_full = ((writeTailptr < InputNum.U) && (writeHeadptr >= InputNum.U)) ||
    ((writeTailptr >= InputNum.U) && (writeHeadptr < InputNum.U))
  when(writeCounter === iter_reg - 1.U) {
    start_write := false.B
  }.elsewhen(array_full && !start_write) {
    start_write    := true.B
    wreg           := waddr_reg
    write_iter_reg := iter_reg
  }.otherwise {
    start_write := start_write
  }

  when(start_write) {
    io.bankWrite(0).io.req.valid     := true.B
    io.bankWrite(0).io.req.bits.addr := wreg + writeCounter
    io.bankWrite(0).io.req.bits.data := Mux(
      writeCounter(4) === 0.U,
      Cat((0 until InputNum).reverse.map(i => regArray(i)(writeCounter(3, 0)))),
      Cat((0 until InputNum).reverse.map(i => regArray(i + InputNum)(writeCounter(3, 0))))
    )
    io.bankWrite(0).io.req.bits.mask := VecInit(Seq.fill(b.memDomain.bankMaskLen)(~0.U(1.W)))
    writeCounter                     := Mux(writeCounter === write_iter_reg - 1.U, 0.U, writeCounter + 1.U)
    writeTailptr                     := Mux(writeTailptr === 2.U * InputNum.U - 1.U, 0.U, writeTailptr + 1.U)
  }
  // Status signals
  io.status.idle    := state === idle
  io.status.running := state === compute
}
