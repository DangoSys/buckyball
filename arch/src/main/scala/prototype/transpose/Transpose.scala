package prototype.transpose

import chisel3._
import chisel3.util._
import chisel3.stage._
import org.chipsalliance.cde.config.Parameters

import prototype.vector._
import framework.builtin.memdomain.mem.{SramReadIO, SramWriteIO}
import examples.toy.balldomain.rs.{BallRsIssue, BallRsComplete}
import examples.BuckyBallConfigs.CustomBuckyBallConfig

class PipelinedTransposer[T <: Data](implicit b: CustomBuckyBallConfig, p: Parameters) extends Module {
  val spad_w = b.veclane * b.inputType.getWidth

  val io = IO(new Bundle {
    // cmd接口
    val cmdReq  = Flipped(Decoupled(new BallRsIssue))
    val cmdResp = Decoupled(new BallRsComplete)

    // 连接到Scratchpad的SRAM读写接口
    val sramRead  = Vec(b.sp_banks, Flipped(new SramReadIO(b.spad_bank_entries, spad_w)))
    val sramWrite = Vec(b.sp_banks, Flipped(new SramWriteIO(b.spad_bank_entries, spad_w, b.spad_mask_len)))
  })

  val idle :: sRead :: sWrite :: complete :: Nil = Enum(4)
  val state = RegInit(idle)

  // 矩阵存储寄存器 (veclane x veclane)
  val regArray = Reg(Vec(b.veclane, Vec(b.veclane, UInt(b.inputType.getWidth.W))))

  // 计数器
  val readCounter  = RegInit(0.U(log2Ceil(b.veclane + 1).W))
  val respCounter  = RegInit(0.U(log2Ceil(b.veclane + 1).W))
  val writeCounter = RegInit(0.U(log2Ceil(b.veclane + 1).W))

  // 指令寄存器
  val robid_reg = RegInit(0.U(10.W))
  val waddr_reg = RegInit(0.U(10.W))
  val wbank_reg = RegInit(0.U(log2Up(b.sp_banks).W))
  val raddr_reg = RegInit(0.U(10.W))
  val rbank_reg = RegInit(0.U(log2Up(b.sp_banks).W))
  val iter_reg  = RegInit(0.U(10.W))

  // 预计算写入数据
  val writeDataReg = Reg(UInt(spad_w.W))
  val writeMaskReg = Reg(Vec(b.spad_mask_len, UInt(1.W)))

  // SRAM默认赋值
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

  // cmd接口默认赋值
  io.cmdReq.ready        := state === idle
  io.cmdResp.valid       := false.B
  io.cmdResp.bits.rob_id := robid_reg

  // 状态机
  switch(state) {
    is(idle) {
      // 指令到达，初始化各个寄存器
      when(io.cmdReq.fire) {
        state        := sRead
        readCounter  := 0.U
        writeCounter := 0.U
        
        robid_reg := io.cmdReq.bits.rob_id
        waddr_reg := io.cmdReq.bits.cmd.op2_bank_addr
        wbank_reg := io.cmdReq.bits.cmd.op2_bank
        raddr_reg := io.cmdReq.bits.cmd.op1_bank_addr
        rbank_reg := io.cmdReq.bits.cmd.op1_bank
        iter_reg  := io.cmdReq.bits.cmd.iter
      }
    }

    is(sRead) {
      when(readCounter < b.veclane.U+1.U) {
      // 发起读取请求
      readCounter := readCounter + 1.U
      io.sramRead(rbank_reg).req.valid     := readCounter < iter_reg
      io.sramRead(rbank_reg).req.bits.addr := raddr_reg + readCounter

      // 准备接收响应
      io.sramRead(rbank_reg).resp.ready := true.B

      // 当收到响应时，存储数据并增加计数器
      val dataWord = io.sramRead(rbank_reg).resp.bits.data

      // 按行写入寄存器阵列
      when(io.sramRead(rbank_reg).resp.fire) {
          for (col <- 0 until b.veclane) {
            val hi = (col + 1) * b.inputType.getWidth - 1
            val lo = col * b.inputType.getWidth
            regArray(respCounter)(col) := dataWord(hi, lo)
      }
      respCounter := respCounter + 1.U
      }
        
    }
        // 如果读取完成，转到写入状态
        when(respCounter  === iter_reg) {
          state := sWrite

          // 预计算第一列写入数据（取转置后第0列 -> 原矩阵每行的第0个元素）
          for (i <- 0 until b.veclane) {
            writeDataReg := Cat((0 until b.veclane).reverse.map(i => regArray(i)(0)))
          }
          // 设置写入掩码（全写）
          for (i <- 0 until b.spad_mask_len) {
            writeMaskReg(i) := 1.U(1.W)
          }
        }
    }

is(sWrite) {
  // 发起写入请求
  io.sramWrite(wbank_reg).req.valid     := writeCounter < b.veclane.U
  io.sramWrite(wbank_reg).req.bits.addr := waddr_reg + writeCounter
  io.sramWrite(wbank_reg).req.bits.data := writeDataReg
  io.sramWrite(wbank_reg).req.bits.mask := writeMaskReg

  // 写入完成，转到完成状态
  when(writeCounter === (b.veclane - 1).U) {
      state := complete
    }.otherwise {
      writeCounter := writeCounter + 1.U
      writeDataReg := Cat((0 until b.veclane).reverse.map(i => regArray(i)(writeCounter + 1.U)))
    }
}

    is(complete) {
      io.cmdResp.valid       := true.B
      io.cmdResp.bits.rob_id := robid_reg

      when(io.cmdResp.fire) {
        state := idle
      }
    }
  }

  // 初始化寄存器阵列
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
