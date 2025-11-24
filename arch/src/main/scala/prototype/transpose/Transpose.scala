package prototype.transpose

import chisel3._
import chisel3.util._
import chisel3.stage._
import org.chipsalliance.cde.config.Parameters

import prototype.vector._
import framework.builtin.memdomain.mem.{SramReadIO, SramWriteIO}
import framework.builtin.frontend.rs.{BallRsIssue, BallRsComplete}
import examples.BuckyballConfigs.CustomBuckyballConfig
import framework.blink.Status
import freechips.rocketchip.tilelink.MemoryOpCategories.wr
import os.read

class PipelinedTransposer[T <: Data](implicit b: CustomBuckyballConfig, p: Parameters) extends Module {
  val spad_w = b.veclane * b.inputType.getWidth

  val io = IO(new Bundle {
    // cmd interface
    val cmdReq  = Flipped(Decoupled(new BallRsIssue))
    val cmdResp = Decoupled(new BallRsComplete)

    // Connect to Scratchpad SRAM read/write interface
    val sramRead  = Vec(b.sp_banks, Flipped(new SramReadIO(b.spad_bank_entries, spad_w)))
    val sramWrite = Vec(b.sp_banks, Flipped(new SramWriteIO(b.spad_bank_entries, spad_w, b.spad_mask_len)))

    // Status output
    val status = new Status
  })

  val idle :: compute :: Nil = Enum(2)
  val state = RegInit(idle)

  // Matrix storage register (veclane x veclane)
  val regArray = Reg(Vec(b.veclane * 2, Vec(b.veclane, UInt(b.inputType.getWidth.W))))

  // Counters
  val readCounter  = RegInit(0.U(10.W))
  val respCounter  = RegInit(0.U(10.W))
  val writeCounter = RegInit(0.U(10.W))
  val respWaitcounter = RegInit(0.U(10.W))
  val writeHeadptr = RegInit(0.U(10.W))
  val writeTailptr = RegInit(0.U(10.W))

  // Instruction registers
  val robid_reg = RegInit(0.U(10.W))
  val waddr_reg = RegInit(0.U(10.W))
  val wbank_reg = RegInit(0.U(log2Up(b.sp_banks).W))
  val raddr_reg = RegInit(0.U(10.W))
  val rbank_reg = RegInit(0.U(log2Up(b.sp_banks).W))
  val iter_reg  = RegInit(0.U(10.W))
  val write_iter_reg = RegInit(0.U(10.W))
  val mode_reg  = RegInit(0.U(1.W))


  // Precompute write data
  val writeDataReg = Reg(UInt(spad_w.W))
  val writeMaskReg = Reg(Vec(b.spad_mask_len, UInt(1.W)))

  val start_write = RegInit(false.B)
  // SRAM default assignment
  for (i <- 0 until b.sp_banks) {
    io.sramRead(i).req.valid        := false.B
    io.sramRead(i).req.bits.addr    := 0.U
    io.sramRead(i).req.bits.fromDMA := false.B
    io.sramRead(i).resp.ready       := false.B

    io.sramWrite(i).req.valid       := false.B
    io.sramWrite(i).req.bits.addr   := 0.U
    io.sramWrite(i).req.bits.data   := 0.U
    io.sramWrite(i).req.bits.mask   := VecInit(Seq.fill(b.spad_mask_len)(0.U(1.W)))
  }

  // cmd interface default assignment
  io.cmdReq.ready        := state === idle

  when(state === idle && io.cmdReq.fire){
    state        := compute
    readCounter  := 0.U
    respCounter  := 0.U
    respWaitcounter := io.cmdReq.bits.cmd.iter + respWaitcounter

    robid_reg := io.cmdReq.bits.rob_id
    waddr_reg := io.cmdReq.bits.cmd.op2_bank_addr
    wbank_reg := io.cmdReq.bits.cmd.op2_bank
    raddr_reg := io.cmdReq.bits.cmd.op1_bank_addr
    rbank_reg := io.cmdReq.bits.cmd.op1_bank
    iter_reg  := io.cmdReq.bits.cmd.iter
    mode_reg  := io.cmdReq.bits.cmd.special(0)
  }
  // read req
  when(((mode_reg === 1.U) &&(state === compute) && RegNext(io.sramWrite(0).req.ready))||
        ((mode_reg === 0.U) && (state === compute))){
      readCounter :=  readCounter + 1.U
      io.sramRead(rbank_reg).req.valid     := readCounter < iter_reg
      io.sramRead(rbank_reg).req.bits.addr := raddr_reg + readCounter
      state := Mux((readCounter >= iter_reg - 1.U) && io.cmdResp.ready, idle, state)
  }
  io.cmdResp.valid       := (readCounter >= (iter_reg - 1.U)) && (state === compute)
  io.cmdResp.bits.rob_id := robid_reg

  // read resp
  io.sramRead(rbank_reg).resp.ready := true.B
  val dataWord = io.sramRead(rbank_reg).resp.bits.data
  val row = respCounter(4,0)
  when(io.sramRead(rbank_reg).resp.fire && respWaitcounter > 0.U){
    for (col <- 0 until b.veclane) {
      val hi = (col + 1) * b.inputType.getWidth - 1
      val lo = col * b.inputType.getWidth
      regArray(row)(col) := dataWord(hi, lo)
    }
    respCounter := Mux(respCounter === iter_reg - 1.U, 0.U, respCounter + 1.U)
    writeHeadptr := Mux(writeHeadptr === 2.U* b.veclane.U - 1.U, 0.U, writeHeadptr + 1.U)
    respWaitcounter := Mux(state === idle && io.cmdReq.fire, io.cmdReq.bits.cmd.iter + respWaitcounter - 1.U, respWaitcounter - 1.U)
  }

  // write req
  val wreg = RegInit(0.U(10.W))
  val array_full = ((writeTailptr < b.veclane.U) && (writeHeadptr >= b.veclane.U)) ||
                   ((writeTailptr >= b.veclane.U) && (writeHeadptr < b.veclane.U))
  when(writeCounter === iter_reg - 1.U){
    start_write := false.B
  }.elsewhen( array_full && !start_write){
    start_write := true.B
    wreg := waddr_reg
    write_iter_reg := iter_reg
  }.otherwise{
    start_write := start_write
  }

  when(start_write){
    io.sramWrite(wbank_reg).req.valid     := true.B
    io.sramWrite(wbank_reg).req.bits.addr := wreg + writeCounter
    io.sramWrite(wbank_reg).req.bits.data := Mux( writeCounter(4) === 0.U, Cat((0 until b.veclane).reverse.map(i => regArray(i)(writeCounter(3,0)))) ,
                                              Cat((0 until b.veclane).reverse.map(i => regArray(i + b.veclane)(writeCounter(3,0)))))
    io.sramWrite(wbank_reg).req.bits.mask := VecInit(Seq.fill(b.spad_mask_len)(~0.U(1.W)))
    writeCounter :=  Mux(writeCounter === write_iter_reg - 1.U, 0.U,writeCounter + 1.U)
    writeTailptr := Mux(writeTailptr === 2.U* b.veclane.U - 1.U, 0.U, writeTailptr + 1.U)
  }
    // Status signals
  io.status.ready := io.cmdReq.ready
  io.status.valid := io.cmdResp.valid
  io.status.idle := state === idle
  io.status.init := readCounter === 0.U && state === compute
  io.status.running := state === compute
  io.status.iter := readCounter
  io.status.complete := io.cmdResp.valid

}
