package prototype.relu

import chisel3._
import chisel3.util._
import chisel3.stage._
import org.chipsalliance.cde.config.Parameters

import prototype.vector._
import framework.memdomain.mem.{SramReadIO, SramWriteIO}
import framework.balldomain.rs.{BallRsIssue, BallRsComplete}
import examples.BuckyballConfigs.CustomBuckyballConfig
import framework.balldomain.blink.Status

class PipelinedRelu[T <: Data](implicit b: CustomBuckyballConfig, p: Parameters)
    extends Module {
  val spad_w = b.veclane * b.inputType.getWidth

  val io = IO(new Bundle {
    // cmd interface
    val cmdReq = Flipped(Decoupled(new BallRsIssue))
    val cmdResp = Decoupled(new BallRsComplete)

    // Connect to Scratchpad SRAM read/write interface
    val sramRead =
      Vec(b.sp_banks, Flipped(new SramReadIO(b.spad_bank_entries, spad_w)))
    val sramWrite = Vec(
      b.sp_banks,
      Flipped(new SramWriteIO(b.spad_bank_entries, spad_w, b.spad_mask_len))
    )

    // Status output
    val status = new Status
  })

  val idle :: sRead :: sWrite :: complete :: Nil = Enum(4)
  val state = RegInit(idle)

  // Store a row of split elements (veclane elements)
  // Store a veclane x veclane tile (perform element-wise ReLU then write back)
  val regArray = RegInit(
    VecInit(Seq.fill(b.veclane)(
      VecInit(Seq.fill(b.veclane)(0.U(b.inputType.getWidth.W)))
    ))
  )

  // Counters
  val readCounter = RegInit(0.U(log2Ceil(b.veclane + 1).W))
  val respCounter = RegInit(0.U(log2Ceil(b.veclane + 1).W))
  val writeCounter = RegInit(0.U(log2Ceil(b.veclane + 1).W))

  // Instruction registers
  val robid_reg = RegInit(0.U(10.W))
  val waddr_reg = RegInit(0.U(10.W))
  val wbank_reg = RegInit(0.U(log2Up(b.sp_banks).W))
  val raddr_reg = RegInit(0.U(10.W))
  val rbank_reg = RegInit(0.U(log2Up(b.sp_banks).W))
  val iter_reg = RegInit(0.U(10.W))
  val cycle_reg = RegInit(0.U(6.W))
  // Batch iteration counter
  val iterCnt = RegInit(0.U(32.W))

  // Precompute write data
  val writeDataReg = Reg(UInt(spad_w.W))
  val writeMaskReg = Reg(Vec(b.spad_mask_len, UInt(1.W)))

  // SRAM default assignment
  for (i <- 0 until b.sp_banks) {
    io.sramRead(i).req.valid          := false.B
    io.sramRead(i).req.bits.addr      := 0.U
    io.sramRead(i).req.bits.fromDMA   := false.B
    io.sramRead(i).resp.ready         := false.B

    io.sramWrite(i).req.valid         := false.B
    io.sramWrite(i).req.bits.addr     := 0.U
    io.sramWrite(i).req.bits.data     := 0.U
    io.sramWrite(i).req.bits.mask     := VecInit(Seq.fill(b.spad_mask_len)(0.U(1.W)))
  }

  // cmd interface default assignment
  io.cmdReq.ready := state === idle
  io.cmdResp.valid := false.B
  io.cmdResp.bits.rob_id := robid_reg

  // State machine
  switch(state) {
    is(idle) {
      when(io.cmdReq.fire) {
        state := sRead
        readCounter := 0.U
        respCounter := 0.U
        writeCounter := 0.U

        robid_reg := io.cmdReq.bits.rob_id
  // For ReLU, output write-back should use decoded wr_bank/addr, not op2_* fields
        waddr_reg := io.cmdReq.bits.cmd.wr_bank_addr
        wbank_reg := io.cmdReq.bits.cmd.wr_bank
        raddr_reg := io.cmdReq.bits.cmd.op1_bank_addr
        rbank_reg := io.cmdReq.bits.cmd.op1_bank
        iter_reg := io.cmdReq.bits.cmd.iter
        cycle_reg := (io.cmdReq.bits.cmd.iter +& (b.veclane.U - 1.U)) / b.veclane.U - 1.U
      }

      when(cycle_reg =/= 0.U) {
        state := sRead
        readCounter := 0.U
        writeCounter := 0.U
        respCounter := 0.U
        waddr_reg := waddr_reg + b.veclane.U
        raddr_reg := raddr_reg + b.veclane.U
        cycle_reg := cycle_reg - 1.U
      }
    }

    is(sRead) {
      when(readCounter < b.veclane.U) {
      // Issue read request
      readCounter := readCounter + 1.U
      io.sramRead(rbank_reg).req.valid     := true.B
      io.sramRead(rbank_reg).req.bits.addr := raddr_reg + readCounter

    }

      // Receive response, only raise ready when there are outstanding reads
      val dataWord = io.sramRead(rbank_reg).resp.bits.data
      // val hasOutstandingRead = readCounter =/= respCounter
      io.sramRead(rbank_reg).resp.ready := true.B
      when(io.sramRead(rbank_reg).resp.fire) {
        for (col <- 0 until b.veclane) {
          val hi = (col + 1) * b.inputType.getWidth - 1
          val lo = col * b.inputType.getWidth
          val raw = dataWord(hi, lo)
          val signed = raw.asSInt
          val relu = Mux(signed < 0.S, 0.S(b.inputType.getWidth.W), signed)
          regArray(respCounter)(col) := relu.asUInt
        }
        respCounter := respCounter + 1.U
      }

      when(respCounter === b.veclane.U) {
        state := sWrite
        // Precompute first write data (row 0, concatenated by column)
        writeDataReg := Cat((0 until b.veclane).reverse.map(j => regArray(0)(j)))
        // Set write mask (write all)
        for (i <- 0 until b.spad_mask_len) {
          writeMaskReg(i) := 1.U(1.W)
        }
      }
    }

    is(sWrite) {
      // Correctly use ready/valid handshake to advance writes, avoid dropped writes
      io.sramWrite(wbank_reg).req.valid := writeCounter < b.veclane.U
      io.sramWrite(wbank_reg).req.bits.addr := waddr_reg + writeCounter
      io.sramWrite(wbank_reg).req.bits.data := writeDataReg
      io.sramWrite(wbank_reg).req.bits.mask := writeMaskReg

      when(writeCounter === (b.veclane - 1).U) {
        state := complete
      }.otherwise {
        writeCounter := writeCounter + 1.U
        // Prepare next row's write data
        writeDataReg := Cat((0 until b.veclane).reverse.map(j => regArray(writeCounter + 1.U)(j)))
      }

    }

    is(complete) {
      when(cycle_reg === 0.U) {
        io.cmdResp.valid        := true.B
        io.cmdResp.bits.rob_id  := robid_reg
        when(io.cmdResp.fire) {
          iterCnt := iterCnt + 1.U
        }
      }
      state := idle
    }
  }

  // Status signals
  io.status.ready := io.cmdReq.ready
  io.status.valid := io.cmdResp.valid
  io.status.idle := (state === idle)
  io.status.init := (state === sRead) && (respCounter < b.veclane.U)
  io.status.running := (state === sWrite) || ((state === sRead) && (respCounter === b.veclane.U))
  io.status.complete := (state === complete) && io.cmdResp.fire
  io.status.iter := iterCnt

  when(reset.asBool) {
    for (i <- 0 until b.veclane) {
      for (j <- 0 until b.veclane) {
        regArray(i)(j) := 0.U
      }
    }
    writeDataReg := 0.U
    for (i <- 0 until b.spad_mask_len) {
      writeMaskReg(i) := 0.U
    }
  }
}
