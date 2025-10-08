package prototype.relu  
  
import chisel3._  
import chisel3.util._  
import org.chipsalliance.cde.config.Parameters  
  
import framework.builtin.memdomain.mem.{SramReadIO, SramWriteIO}  
import framework.builtin.frontend.rs.{BallRsIssue, BallRsComplete}  
import examples.BuckyBallConfigs.CustomBuckyBallConfig  
  
class ReluAccelerator(implicit b: CustomBuckyBallConfig, p: Parameters)  
    extends Module {  
  val spad_w = b.veclane * b.inputType.getWidth  
  
  val io = IO(new Bundle {  
    // 命令接口  
    val cmdReq = Flipped(Decoupled(new BallRsIssue))  
    val cmdResp = Decoupled(new BallRsComplete)  
  
    // Scratchpad读写接口  
    val sramRead =  
      Vec(b.sp_banks, Flipped(new SramReadIO(b.spad_bank_entries, spad_w)))  
    val sramWrite = Vec(  
      b.sp_banks,  
      Flipped(new SramWriteIO(b.spad_bank_entries, spad_w, b.spad_mask_len))  
    )  
  })  
  
  // 状态机定义  
  val idle :: read :: compute :: write :: complete :: Nil = Enum(5)  
  val state = RegInit(idle)  
  
  // 寄存器定义  
  val robid_reg = RegInit(0.U(10.W))  
  val raddr_reg = RegInit(0.U(10.W))  
  val rbank_reg = RegInit(0.U(log2Up(b.sp_banks).W))  
  val waddr_reg = RegInit(0.U(10.W))  
  val wbank_reg = RegInit(0.U(log2Up(b.sp_banks).W))  
  val iter_reg = RegInit(0.U(10.W))  
  val counter = RegInit(0.U(10.W))  
  
  // 数据缓冲区  
  val dataBuffer = Reg(Vec(b.veclane, UInt(b.inputType.getWidth.W)))  
  
  // SRAM默认赋值  
  for (i <- 0 until b.sp_banks) {  
    io.sramRead(i).req.valid := false.B  
    io.sramRead(i).req.bits.addr := 0.U  
    io.sramRead(i).req.bits.fromDMA := false.B  
    io.sramRead(i).resp.ready := false.B  
    io.sramWrite(i).req.valid := false.B  
    io.sramWrite(i).req.bits.addr := 0.U  
    io.sramWrite(i).req.bits.data := 0.U  
    io.sramWrite(i).req.bits.mask := VecInit(  
      Seq.fill(b.spad_mask_len)(0.U(1.W))  
    )  
  }  
  
  // 命令接口默认赋值  
  io.cmdReq.ready := state === idle  
  io.cmdResp.valid := false.B  
  io.cmdResp.bits.rob_id := robid_reg  
  
  // 状态机逻辑  
  switch(state) {  
    is(idle) {  
      when(io.cmdReq.fire) {  
        robid_reg := io.cmdReq.bits.rob_id  
        raddr_reg := io.cmdReq.bits.cmd.op1_bank_addr  
        rbank_reg := io.cmdReq.bits.cmd.op1_bank  
        waddr_reg := io.cmdReq.bits.cmd.wr_bank_addr  
        wbank_reg := io.cmdReq.bits.cmd.wr_bank  
        iter_reg := io.cmdReq.bits.cmd.iter  
        counter := 0.U  
        state := read  
      }  
    }  
  
    is(read) {  
      // 发起读请求  
      io.sramRead(rbank_reg).req.valid := true.B  
      io.sramRead(rbank_reg).req.bits.addr := raddr_reg + counter  
      io.sramRead(rbank_reg).resp.ready := true.B  
  
      when(io.sramRead(rbank_reg).resp.fire) {  
        // 将读取的数据存入缓冲区  
        for (i <- 0 until b.veclane) {  
          dataBuffer(i) := io  
            .sramRead(rbank_reg)  
            .resp  
            .bits  
            .data((i + 1) * b.inputType.getWidth - 1, i * b.inputType.getWidth)  
        }  
        state := compute  
      }  
    }  
  
    is(compute) {  
      // ReLU计算：max(0, x)  
      for (i <- 0 until b.veclane) {  
        when(dataBuffer(i).asSInt < 0.S) {  
          dataBuffer(i) := 0.U  
        }  
      }  
      state := write  
    }  
  
    is(write) {  
      // 写回结果  
      io.sramWrite(wbank_reg).req.valid := true.B  
      io.sramWrite(wbank_reg).req.bits.addr := waddr_reg + counter  
      io.sramWrite(wbank_reg).req.bits.data := Cat(dataBuffer.reverse)  
      io.sramWrite(wbank_reg).req.bits.mask := VecInit(  
        Seq.fill(b.spad_mask_len)(1.U(1.W))  
      )  
  
      when(io.sramWrite(wbank_reg).req.fire) {  
        // 修正：先检查是否完成，再递增计数器  
        when(counter === iter_reg - 1.U) {  
          state := complete  
        }.otherwise {  
          counter := counter + 1.U  
          state := read  
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
}

class ReluUnit(implicit b: CustomBuckyBallConfig, p: Parameters) extends Module {  
  val spad_w = b.veclane * b.inputType.getWidth  
    
  val io = IO(new Bundle {  
    val cmdReq = Flipped(Decoupled(new BallRsIssue))  
    val cmdResp = Decoupled(new BallRsComplete)  
    val sramRead = Vec(b.sp_banks, Flipped(new SramReadIO(b.spad_bank_entries, spad_w)))  
    val sramWrite = Vec(b.sp_banks, Flipped(new SramWriteIO(b.spad_bank_entries, spad_w, b.spad_mask_len)))  
  })  
    
  // 实现 ReLU 逻辑:读取数据,应用 max(0, x),写回  
}