package prototype.matrix

import chisel3._
import chisel3.util._
import chisel3.stage._
import org.chipsalliance.cde.config.Parameters

import prototype.matrix._
import framework.memdomain.mem.{SramWriteIO, SramReadIO, SramReadResp}
import examples.BuckyballConfigs.CustomBuckyballConfig
import framework.balldomain.rs.{BallRsIssue, BallRsComplete}

class BBFP_EX(implicit b: CustomBuckyballConfig, p: Parameters) extends Module {
  val rob_id_width = log2Up(b.rob_entries)
  val spad_w = b.veclane * b.inputType.getWidth

  val io = IO(new Bundle {
    val sramWrite = Vec(b.sp_banks, Flipped(new SramWriteIO(b.spad_bank_entries, spad_w, b.spad_mask_len)))
    val lu_ex_i = Flipped(Decoupled(new lu_ex_req))
    val sramReadResp = Vec(b.sp_banks, Flipped(Decoupled(new SramReadResp(spad_w))))
    val is_matmul_ws = Input(Bool())
    val accWrite = Vec(b.acc_banks, Flipped(new SramWriteIO(b.acc_bank_entries, b.acc_w, b.acc_mask_len)))
    val cmdResp = Decoupled(new BallRsComplete)
  })

     for(i <- 0 until b.sp_banks) {
    io.sramWrite(i).req.valid := false.B
    io.sramWrite(i).req.bits.addr := 0.U
    io.sramWrite(i).req.bits.data := 0.U
    io.sramWrite(i).req.bits.mask := VecInit(Seq.fill(spad_w / 8)(false.B))
  }

   for(i <- 0 until b.acc_banks) {
    io.accWrite(i).req.valid := false.B
    io.accWrite(i).req.bits.addr := DontCare
    io.accWrite(i).req.bits.data := DontCare
    io.accWrite(i).req.bits.mask := VecInit(Seq.fill(b.acc_mask_len)(true.B))
  }

  io.cmdResp.valid := false.B
  io.cmdResp.bits.rob_id := 0.U

  val idle::weight_load::data_compute::Nil = Enum(3)
  val weight_cycles = RegInit(0.U(10.W))
  val act_cycles = RegInit(0.U(10.W))
  val weight_expreg=RegInit(0.U(32.W))
  val act_expreg=RegInit(0.U(32.W))
  // Extract signals from pipeline frontend
  val op1_bank_reg  = Reg(UInt(io.lu_ex_i.bits.op1_bank.getWidth.W))
  val op2_bank_reg  = Reg(UInt(io.lu_ex_i.bits.op2_bank.getWidth.W))
  val wr_bank   = Reg(UInt(io.lu_ex_i.bits.wr_bank.getWidth.W))
  val opcode    = Reg(UInt(io.lu_ex_i.bits.opcode.getWidth.W))


  val act_shift_reg = Reg(Vec(16, Vec(16, UInt(7.W))))
  val col_enable = RegInit(VecInit(Seq.fill(16)(false.B)))
  val input_cycle = RegInit(0.U(5.W))
  val act_data_ready = RegInit(false.B)


  when(io.lu_ex_i.valid) {
    op1_bank_reg  := io.lu_ex_i.bits.op1_bank
    op2_bank_reg  := io.lu_ex_i.bits.op2_bank
    wr_bank   := io.lu_ex_i.bits.wr_bank
    opcode  := io.lu_ex_i.bits.opcode
  }

  val op1_bank = Mux(io.lu_ex_i.valid, io.lu_ex_i.bits.op1_bank, op1_bank_reg)
  val op2_bank = Mux(io.lu_ex_i.valid, io.lu_ex_i.bits.op2_bank, op2_bank_reg)

  val state = RegInit(idle)
  val pe_array = Module(new BBFP_PE_Array16x16)

  // Use shift registers instead of original regular registers


  // Activation data input logic
  val act_reg_ptr = RegInit(0.U(5.W))

  // In idle and weight_load phases, store data like regular registers
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

  // New registers for weight and activation counting

  // Save 64 4x32 registers for output
  val output_buffer = Reg(Vec(64, Vec(4, UInt(32.W))))
  // Save parallelogram output
  val output_buffer_parallelogram = Reg(Vec(32, Vec(16, UInt(32.W))))
  val output_ptr = RegInit(0.U(5.W))
  val output_ready = RegInit(false.B)

  // New output data processing logic
  // Extended to 7 bits to support 64 cycles
  val write_cycles = RegInit(0.U(7.W))
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



  // PE array default value assignment
  pe_array.io.in_last := VecInit(Seq.fill(16)(false.B))
  pe_array.io.in_id := DontCare
  pe_array.io.in_a := DontCare
  pe_array.io.in_d := DontCare
  pe_array.io.in_b := VecInit(Seq.fill(16)(0.U))
  pe_array.io.in_control.foreach(_.propagate := 0.U)
  pe_array.io.in_valid := VecInit(Seq.fill(16)(false.B))

  // Default SRAM write port assignment
  for(i <- 0 until b.sp_banks){
    io.sramWrite(i).req.valid := false.B
    io.sramWrite(i).req.bits.addr := 0.U
    io.sramWrite(i).req.bits.data := 0.U
    io.sramWrite(i).req.bits.mask := VecInit(Seq.fill(spad_w / 8)(false.B))
  }

  // Start writing to SRAM when output is ready
  when(output_ready && !writing_output) {
    writing_output := true.B
    write_cycles := 0.U
  }

  when(writing_output) {
    when(write_cycles < 16.U) {
    // Concatenate 4x32-bit data into 128-bit wide data and write to SRAM
     for(i <- 0 until b.acc_banks/2) {
      when((wr_bank_addr_base + write_cycles)(0) === 0.U){
        io.accWrite(i).req.valid := true.B
        io.accWrite(i).req.bits.addr := wr_bank_addr_base + (write_cycles >> 1.U)
        val idx = (write_cycles * 4.U + i.U)(5,0) // 6 bits for 64 elements
        io.accWrite(i).req.bits.data := Cat(
          output_buffer(idx)(3),
          output_buffer(idx)(2),
          output_buffer(idx)(1),
          output_buffer(idx)(0)
        )
        io.accWrite(i).req.bits.mask := VecInit(Seq.fill(b.acc_mask_len)(true.B))
      }.otherwise{
        io.accWrite(i + b.acc_banks/2).req.valid := true.B
        io.accWrite(i + b.acc_banks/2).req.bits.addr := wr_bank_addr_base + (write_cycles >> 1.U)
        val idx = (write_cycles * 4.U + i.U)(5,0) // 6 bits for 64 elements
        io.accWrite(i + b.acc_banks/2).req.bits.data := Cat(
          output_buffer(idx)(3),
          output_buffer(idx)(2),
          output_buffer(idx)(1),
          output_buffer(idx)(0)
        )
        io.accWrite(i + b.acc_banks/2).req.bits.mask := VecInit(Seq.fill(b.acc_mask_len)(true.B))
      }
    }
    write_cycles := write_cycles + 1.U
    }.otherwise {
    // Write completed
      writing_output := false.B
      output_ready := false.B
      addr_base_captured:=false.B
      io.cmdResp.bits.rob_id := io.lu_ex_i.bits.rob_id
      io.cmdResp.valid := true.B
    }
  }

  io.lu_ex_i.ready := true.B

  switch(state) {
    is(idle) {
    when(io.lu_ex_i.valid && !io.is_matmul_ws) {
      // Start weight loading
      weight_cycles := 0.U
      state := weight_load
    }.elsewhen(io.is_matmul_ws && act_data_ready){
      state := data_compute
      act_cycles := 0.U
    }
    }
    is(weight_load) {
    // Load 16 cycles of weights
    when(weight_cycles < 16.U) {
      pe_array.io.in_d := weight_reg
      pe_array.io.in_control.foreach(_.propagate := 1.U)
      pe_array.io.in_valid.foreach(_ := true.B)
      weight_cycles := weight_cycles + 1.U
    }.otherwise {
      // Weight loading completed, wait for activation data to be ready then enter computation
      when(act_data_ready) {
      act_cycles := 0.U
      state := data_compute
      }
    }
    }
    is(data_compute) {
    act_cycles := act_cycles + 1.U
    // In computation phase, start parallelogram input mode for shift registers
    when(act_cycles < 31.U) {
      // Each cycle enables more rows for shifting row by row
      for(col <- 0 until 16) {
      when(col.U <= act_cycles && col.U < 16.U) {
        col_enable(col) := true.B
      }.otherwise {
        col_enable(col) := false.B
      }
      }

      // Shift register operation: enabled rows shift right
      for(col <- 0 until 16) {
      when(col_enable(col)) {
        for(row <- 0 until 15) {
        act_shift_reg(row)(col) := act_shift_reg(row + 1)(col)
        }
        // Pad left with 0 (because in computation phase, no new data is input)
        act_shift_reg(15)(col) := 0.U(7.W)
      }
      }

      // Input rightmost element of each row (column 15) to PE array
      val current_input = WireDefault(VecInit(Seq.fill(16)(0.U(7.W))))
      for(col <- 0 until 16) {
      when(col_enable(col)) {
        current_input(col) := act_shift_reg(0)(col)
      }.otherwise {
        current_input(col) := 0.U(7.W)
      }
      }
      pe_array.io.in_a := current_input
      pe_array.io.in_b.foreach(_ := 0.U)
    }
    // Start receiving output from cycle 16, end at cycle 47

    when(act_cycles > 16.U && act_cycles <= 48.U) {
      output_buffer_parallelogram(output_ptr) := pe_array.io.out_b
      output_ptr := output_ptr + 1.U
    }

    // After cycle 47, convert parallelogram output to 64 4x32 registers
    when(act_cycles === 49.U) {
      output_ready := true.B
      // Convert parallelogram output to 64 4x32 registers, organized in row-major order
      for(row <- 0 until 16) {
      for(col <- 0 until 16) {
        val src_row = row + col
        // Every 4 columns form one register group
        val buffer_index = row * 4 + (col >> 2)
        // Position within 4-element group
        val element_index = col & 0x3
        if(src_row < 32) {
        output_buffer(buffer_index)(element_index) := output_buffer_parallelogram(src_row)(col)
        } else {
        output_buffer(buffer_index)(element_index) := 0.U(32.W)
        }
      }
      }
    }

    // Reset data ready flag
    when(act_cycles === 50.U) {
      act_data_ready := false.B
      act_reg_ptr := 0.U
      output_ptr := 0.U
      // Reset all row enable signals
      col_enable := VecInit(Seq.fill(16)(false.B))
      state := idle
    }
    }
  }


  io.sramReadResp.foreach { resp =>
    resp.ready := state =/= idle
  }
}
