package prototype.transpose

import chisel3._
import chisel3.util._
import chisel3.stage._
import chisel3.experimental.hierarchy.{instantiable, public}
import chisel3.experimental.{SerializableModule, SerializableModuleParameter}
import org.chipsalliance.cde.config.Parameters

import prototype.vector._
import framework.memdomain.backend.banks.{SramReadIO, SramWriteIO}
import framework.balldomain.rs.{BallRsComplete, BallRsIssue}
import examples.toy.balldomain.BallDomainParam
import framework.balldomain.blink.Status
import prototype.transpose.configs.TransposeConfig

@instantiable
class PipelinedTransposer(val parameter: TransposeConfig)(implicit p: Parameters)
    extends Module
    with SerializableModule[TransposeConfig] {
  // Get parameters from config
  val ballParam  = parameter.ballParam
  val InputNum   = parameter.InputNum
  val inputWidth = parameter.inputWidth
  val bankWidth  = parameter.bankWidth

  @public
  val io = IO(new Bundle {
    // cmd interface
    val cmdReq  = Flipped(Decoupled(new BallRsIssue(ballParam)))
    val cmdResp = Decoupled(new BallRsComplete(ballParam))

    // Connect to unified bank read/write interface
    val bankRead  = Vec(ballParam.numBanks, Flipped(new SramReadIO(ballParam.bankEntries, bankWidth)))
    val bankWrite =
      Vec(ballParam.numBanks, Flipped(new SramWriteIO(ballParam.bankEntries, bankWidth, ballParam.bankMaskLen)))

    // Status output
    val status = new Status
  })

  val idle :: compute :: Nil = Enum(2)
  val state                  = RegInit(idle)

  // Matrix storage register (InputNum x InputNum)
  val regArray = Reg(Vec(InputNum * 2, Vec(InputNum, UInt(inputWidth.W))))

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
  val wbank_reg      = RegInit(0.U(log2Up(ballParam.numBanks).W))
  val raddr_reg      = RegInit(0.U(10.W))
  val rbank_reg      = RegInit(0.U(log2Up(ballParam.numBanks).W))
  val iter_reg       = RegInit(0.U(10.W))
  val write_iter_reg = RegInit(0.U(10.W))
  val mode_reg       = RegInit(0.U(1.W))

  // Precompute write data
  val writeDataReg = Reg(UInt(bankWidth.W))
  val writeMaskReg = Reg(Vec(ballParam.bankMaskLen, UInt(1.W)))

  val start_write = RegInit(false.B)
  // SRAM default assignment
  for (i <- 0 until ballParam.numBanks) {
    io.bankRead(i).req.valid        := false.B
    io.bankRead(i).req.bits.addr    := 0.U
    io.bankRead(i).req.bits.fromDMA := false.B
    io.bankRead(i).resp.ready       := false.B

    io.bankWrite(i).req.valid     := false.B
    io.bankWrite(i).req.bits.addr := 0.U
    io.bankWrite(i).req.bits.data := 0.U
    io.bankWrite(i).req.bits.mask := VecInit(Seq.fill(parameter.bankMaskLen)(0.U(1.W)))
  }

  // cmd interface default assignment
  io.cmdReq.ready := state === idle

  when(state === idle && io.cmdReq.fire) {
    state           := compute
    readCounter     := 0.U
    respCounter     := 0.U
    respWaitcounter := io.cmdReq.bits.cmd.iter + respWaitcounter

    robid_reg := io.cmdReq.bits.rob_id
    waddr_reg := 0.U // New ISA: all operations start from row 0
    wbank_reg := io.cmdReq.bits.cmd.op2_bank
    raddr_reg := 0.U // New ISA: all operations start from row 0
    rbank_reg := io.cmdReq.bits.cmd.op1_bank
    iter_reg  := io.cmdReq.bits.cmd.iter
    mode_reg  := io.cmdReq.bits.cmd.special(0)
  }
  // read req
  when(((mode_reg === 1.U) && (state === compute) && RegNext(io.bankWrite(0).req.ready)) ||
    ((mode_reg === 0.U) && (state === compute))) {
    readCounter                          := readCounter + 1.U
    io.bankRead(rbank_reg).req.valid     := readCounter < iter_reg
    io.bankRead(rbank_reg).req.bits.addr := raddr_reg + readCounter
    state                                := Mux((readCounter >= iter_reg - 1.U) && io.cmdResp.ready, idle, state)
  }
  io.cmdResp.valid       := (readCounter >= (iter_reg - 1.U)) && (state === compute)
  io.cmdResp.bits.rob_id := robid_reg

  // read resp
  io.bankRead(rbank_reg).resp.ready := true.B
  val dataWord = io.bankRead(rbank_reg).resp.bits.data
  val row      = respCounter(4, 0)
  when(io.bankRead(rbank_reg).resp.fire && respWaitcounter > 0.U) {
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
    io.bankWrite(wbank_reg).req.valid     := true.B
    io.bankWrite(wbank_reg).req.bits.addr := wreg + writeCounter
    io.bankWrite(wbank_reg).req.bits.data := Mux(
      writeCounter(4) === 0.U,
      Cat((0 until InputNum).reverse.map(i => regArray(i)(writeCounter(3, 0)))),
      Cat((0 until InputNum).reverse.map(i => regArray(i + InputNum)(writeCounter(3, 0))))
    )
    io.bankWrite(wbank_reg).req.bits.mask := VecInit(Seq.fill(parameter.bankMaskLen)(~0.U(1.W)))
    writeCounter                          := Mux(writeCounter === write_iter_reg - 1.U, 0.U, writeCounter + 1.U)
    writeTailptr                          := Mux(writeTailptr === 2.U * InputNum.U - 1.U, 0.U, writeTailptr + 1.U)
  }
  // Status signals
  io.status.ready    := io.cmdReq.ready
  io.status.valid    := io.cmdResp.valid
  io.status.idle     := state === idle
  io.status.init     := readCounter === 0.U && state === compute
  io.status.running  := state === compute
  io.status.iter     := readCounter
  io.status.complete := io.cmdResp.valid

}
