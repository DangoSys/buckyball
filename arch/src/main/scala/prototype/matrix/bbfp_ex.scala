package prototype.matrix

import chisel3._
import chisel3.util._
import chisel3.stage._
import org.chipsalliance.cde.config.Parameters

import prototype.matrix._
import examples.toy.balldomain.{ExReservationStationIssue, ExReservationStationComplete, ExBuckyBallCmd}
import framework.builtin.mem.{SramReadIO, SramWriteIO, SramReadResp}
import examples.BuckyBallConfigs.CustomBuckyBallConfig

class BBFP_EX(implicit bbconfig: CustomBuckyBallConfig, p: Parameters) extends Module {
    val rob_id_width = log2Up(bbconfig.rob_entries)
    val spad_w = bbconfig.veclane * bbconfig.inputType.getWidth

    val io = IO(new Bundle {
        val sramWrite = Vec(bbconfig.sp_banks, Flipped(new SramWriteIO(bbconfig.spad_bank_entries, spad_w, bbconfig.spad_mask_len)))
        val lu_ex_i = Flipped(Decoupled(new lu_ex_req))
        val sramReadResp = Vec(bbconfig.sp_banks, Flipped(Decoupled(new SramReadResp(spad_w))))
        val is_matmul_ws = Input(Bool())
        val accWrite = Vec(bbconfig.acc_banks, Flipped(new SramWriteIO(bbconfig.acc_bank_entries, bbconfig.acc_w, bbconfig.acc_mask_len)))
  })

       for(i <- 0 until bbconfig.sp_banks) {
        io.sramWrite(i).req.valid := false.B
        io.sramWrite(i).req.bits.addr := 0.U
        io.sramWrite(i).req.bits.data := 0.U
        io.sramWrite(i).req.bits.mask := VecInit(Seq.fill(spad_w / 8)(false.B))
    }

     for(i <- 0 until bbconfig.acc_banks) {
        io.accWrite(i).req.valid := false.B
        io.accWrite(i).req.bits.addr := DontCare
        io.accWrite(i).req.bits.data := DontCare
        io.accWrite(i).req.bits.mask := VecInit(Seq.fill(bbconfig.acc_mask_len)(true.B))
    }
    val idle::weight_load::data_compute::Nil = Enum(3)
    val weight_cycles = RegInit(0.U(10.W))
    val act_cycles = RegInit(0.U(10.W))
    val weight_expreg=RegInit(0.U(32.W))
    val act_expreg=RegInit(0.U(32.W))
    // 提取流水线前端的信号
    val op1_bank    = Reg(UInt(io.lu_ex_i.bits.op1_bank.getWidth.W))
    val op2_bank    = Reg(UInt(io.lu_ex_i.bits.op2_bank.getWidth.W))
    val wr_bank     = Reg(UInt(io.lu_ex_i.bits.wr_bank.getWidth.W))
    val opcode      = Reg(UInt(io.lu_ex_i.bits.opcode.getWidth.W))
 
    
    val act_shift_reg = Reg(Vec(16, Vec(16, UInt(7.W))))
    val row_enable = RegInit(VecInit(Seq.fill(16)(false.B)))
    val input_cycle = RegInit(0.U(5.W))
    val act_data_ready = RegInit(false.B)


    when(io.lu_ex_i.valid) {
        op1_bank  := io.lu_ex_i.bits.op1_bank
        op2_bank  := io.lu_ex_i.bits.op2_bank
        wr_bank   := io.lu_ex_i.bits.wr_bank
        opcode    := io.lu_ex_i.bits.opcode
    }

    val state = RegInit(idle)
    val pe_array = Module(new BBFP_PE_Array16x16)

    // 使用移位寄存器代替原来的普通寄存器
   

    // 激活数据输入逻辑
    val act_reg_ptr = RegInit(0.U(5.W))
    
    // 在idle和weight_load阶段，像普通寄存器一样存储数据
    when(io.sramReadResp(op2_bank).valid && state =/= data_compute && act_data_ready === false.B) {
      val data = io.sramReadResp(op2_bank).bits.data
      for(i <- 0 until 16) {
        act_shift_reg(act_reg_ptr)(i) := data((i+1)*8-1, i*8)(6,0)
      }
      act_reg_ptr := act_reg_ptr + 1.U
    }
    when(act_reg_ptr === 16.U && state =/= data_compute) {
      act_data_ready := true.B
    }
    val weight_reg = Reg(Vec(16, UInt(7.W)))
 
    when(io.sramReadResp(op1_bank).valid && !io.is_matmul_ws) {
      val data = io.sramReadResp(op1_bank).bits.data
      for(i <- 0 until 16) {
        weight_reg(i) := data((i+1)*8-1, i*8)(6,0)
      }
    }

    // 新增寄存器用于权重和激活计数

    val output_buffer = Reg(Vec(64, Vec(4, UInt(32.W)))) // 保存输出的64个4x32寄存器
    val output_buffer_parallelogram = Reg(Vec(32, Vec(16, UInt(32.W)))) // 保存平行四边形输出
    val output_ptr = RegInit(0.U(5.W))
    val output_ready = RegInit(false.B)

    // 新的输出数据处理逻辑
    val write_cycles = RegInit(0.U(7.W)) // 扩展到7位以支持64个周期
    val writing_output = RegInit(false.B)

    val wr_bank_addr_base = Reg(UInt(io.lu_ex_i.bits.wr_bank_addr.getWidth.W))
    val addr_base_captured = RegInit(false.B)

    when(io.lu_ex_i.valid && !addr_base_captured) {
      wr_bank_addr_base := io.lu_ex_i.bits.wr_bank_addr
      addr_base_captured := true.B
      weight_expreg := io.sramReadResp(op1_bank).bits.data(127,96)
      act_expreg := io.sramReadResp(op2_bank).bits.data(127,96)
    }
    val result_exp = weight_expreg+act_expreg

   

    // PE阵列默认值赋值
    pe_array.io.in_last := VecInit(Seq.fill(16)(false.B))
    pe_array.io.in_id := DontCare
    pe_array.io.in_a := DontCare
    pe_array.io.in_d := DontCare
    pe_array.io.in_b := VecInit(Seq.fill(16)(0.U))
    pe_array.io.in_control.foreach(_.propagate := 0.U)
    pe_array.io.in_valid := VecInit(Seq.fill(16)(false.B))

    // 默认SRAM写端口赋值
    for(i <- 0 until bbconfig.sp_banks){
        io.sramWrite(i).req.valid := false.B
        io.sramWrite(i).req.bits.addr := 0.U
        io.sramWrite(i).req.bits.data := 0.U
        io.sramWrite(i).req.bits.mask := VecInit(Seq.fill(spad_w / 8)(false.B))
    }

    // 当输出准备好时，开始写入SRAM
    when(output_ready && !writing_output) {
      writing_output := true.B
      write_cycles := 0.U
    }

    when(writing_output) {
      when(write_cycles < 16.U) {
        // 将一个4x32位数据拼接成128位宽的数据写入SRAM
       
        
         for(i <- 0 until bbconfig.acc_banks) {
        io.accWrite(i).req.valid := true.B
        io.accWrite(i).req.bits.addr := (wr_bank_addr_base >> log2Ceil(bbconfig.acc_banks)) + write_cycles
        val idx = (write_cycles * 4.U + i.U)(5,0) // 6 bits for 64 elements
        io.accWrite(i).req.bits.data := Cat(
          output_buffer(idx)(3),
          output_buffer(idx)(2),
          output_buffer(idx)(1),
          output_buffer(idx)(0)
        )
        io.accWrite(i).req.bits.mask := VecInit(Seq.fill(bbconfig.acc_mask_len)(true.B))
    }
        write_cycles := write_cycles + 1.U
      }.otherwise {                                                                                                                            
        // 写入完成
        writing_output := false.B
        output_ready := false.B
        addr_base_captured:=false.B
      }
    }

    io.lu_ex_i.ready := true.B

    switch(state) {
      is(idle) {
        when(io.lu_ex_i.valid && !io.is_matmul_ws) {
          // 启动权重加载
          weight_cycles := 0.U
          state := weight_load
        }.elsewhen(io.is_matmul_ws && act_data_ready){
            state := data_compute
            act_cycles := 0.U
        }
      }
      is(weight_load) {
        // 加载16周期权重
        when(weight_cycles <= 16.U) {
          pe_array.io.in_d := weight_reg
          pe_array.io.in_control.foreach(_.propagate := 1.U)
          pe_array.io.in_valid.foreach(_ := true.B)
          weight_cycles := weight_cycles + 1.U
        }.otherwise {
          // 权重加载完成，等待激活数据准备好后进入计算
          when(act_data_ready) {
            act_cycles := 0.U
            state := data_compute
          }
        }
      }
      is(data_compute) {
        act_cycles := act_cycles + 1.U
        // 在计算阶段，启动移位寄存器的平行四边形输入模式
        when(act_cycles < 31.U) {
          // 每个周期逐行使能更多行进行移位
          for(row <- 0 until 16) {
            when(row.U <= act_cycles && row.U < 16.U) {
              row_enable(row) := true.B
            }.otherwise {
              row_enable(row) := false.B
            }
          }
        
          // 移位寄存器操作：使能的行向右移位
          for(row <- 0 until 16) {
            when(row_enable(row)) {
              for(col <- 15 to 1 by -1) {
                act_shift_reg(row)(col) := act_shift_reg(row)(col-1)
              }
              // 左侧补0（因为是计算阶段，不再输入新数据）
              act_shift_reg(row)(0) := 0.U(7.W)
            }
          }
                 
          // 将每行的最右侧元素（第15列）输入到PE阵列
          val current_input = WireDefault(VecInit(Seq.fill(16)(0.U(7.W))))
          for(row <- 0 until 16) {
            when(row_enable(row)) {
              current_input(row) := act_shift_reg(row)(15)
            }.otherwise {
              current_input(row) := 0.U(7.W)
            }
          }
          pe_array.io.in_a := current_input
          pe_array.io.in_b.foreach(_ := 0.U)
        }
        // 从第16个周期开始接收输出，到第47个周期结束
        when(act_cycles > 16.U && act_cycles <= 48.U) {
          output_buffer_parallelogram(output_ptr) := pe_array.io.out_b
          output_ptr := output_ptr + 1.U
        }
        
        // 第47个周期后，将平行四边形输出转换为64个4x32寄存器
        when(act_cycles === 49.U) {
          output_ready := true.B
          // 转换平行四边形输出为64个4x32寄存器，按行优先顺序组织
          for(row <- 0 until 16) {
            for(col <- 0 until 16) {
              val src_row = row + col
              val buffer_index = row * 4 + (col >> 2) // 每4列为一个寄存器组
              val element_index = col & 0x3 // 在4元素组内的位置
              if(src_row < 32) {
                output_buffer(buffer_index)(element_index) := output_buffer_parallelogram(src_row)(col)
              } else {
                output_buffer(buffer_index)(element_index) := 0.U(32.W)
              }
            }
          }
        }
        
        // 重置数据准备标志
        when(act_cycles === 50.U) {
          act_data_ready := false.B
          act_reg_ptr := 0.U
          output_ptr := 0.U
          // 重置所有行使能信号
          row_enable := VecInit(Seq.fill(16)(false.B))
          state := idle
        }
      }
    }


    io.sramReadResp.foreach { resp =>
        resp.ready := true.B
    }
}
