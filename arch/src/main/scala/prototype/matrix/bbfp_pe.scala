package prototype.matrix

import chisel3._
import chisel3.util._
import chisel3.stage._
import prototype.matrix._

// PE control signal Bundle
class PEControl extends Bundle {
  // Propagation control
  val propagate = UInt(1.W)
}

class MacUnit extends Module {

  val io = IO(new Bundle {
    // Unsigned input: [6]=sign, [5]=flag, [4:0]=value
    val in_a  = Input(UInt(7.W))
    // Unsigned input: [6]=sign, [5]=flag, [4:0]=value
    val in_b  = Input(UInt(7.W))
    // Unsigned input: [31]=sign, [30:0]=value
    val in_c  = Input(UInt(32.W))
    // Signed output
    val out_d = Output(UInt(32.W))
  })

  // Extract sign bits
  val sign_a = io.in_a(6)
  val sign_b = io.in_b(6)
  val sign_c = io.in_c(31)

  // Extract flag bits
  val flag_a = io.in_a(5)
  val flag_b = io.in_b(5)

  // Extract value parts
  // 5-bit value
  val value_a = io.in_a(4, 0)
  // 5-bit value
  val value_b = io.in_b(4, 0)
  // 31-bit value
  val value_c = io.in_c(30, 0)

  // val extended_a = value_a.asUInt().pad(7) // Extend to 7 bits
  // val extended_b = value_b.asUInt().pad(7) // Extend to 7 bits

  // Determine left shift based on flag bit
  val shifted_a = Mux(flag_a === 1.U, value_a << 2, value_a)
  val shifted_b = Mux(flag_b === 1.U, value_b << 2, value_b)

  // Convert value to signed number, considering sign bit
  // First extend bit width to avoid overflow, then determine sign based on sign bit
  val a_signed = Mux(sign_a === 1.U, -(shifted_a.zext), shifted_a.zext).asSInt

  val b_signed = Mux(sign_b === 1.U, -(shifted_b.zext), shifted_b.zext).asSInt

  val c_signed = Mux(sign_c === 1.U, -(value_c.zext), value_c.zext).asSInt

  // Perform MAC operation: a * b + c
  val result = a_signed * b_signed + c_signed

  // Output result
  io.out_d := result.asUInt
}

// BBFP PE unit (only supports Weight Stationary)
class BBFP_PE_WS(max_simultaneous_matmuls: Int = 16) extends Module {

  val io = IO(new Bundle {
    // Data input/output
    // Input activation value
    val in_a        = Input(UInt(7.W))
    // Input partial sum
    val in_b        = Input(UInt(32.W))
    // Input weight
    val in_d        = Input(UInt(7.W))
    // Output activation value
    val out_a       = Output(UInt(7.W))
    // Output partial sum
    val out_b       = Output(UInt(32.W))
    // Output weight
    val out_d       = Output(UInt(7.W))
    // Control signals
    val in_control  = Input(new PEControl())
    val out_control = Output(new PEControl())

    // ID and valid signals
    val in_id  = Input(UInt(log2Up(max_simultaneous_matmuls).W))
    val out_id = Output(UInt(log2Up(max_simultaneous_matmuls).W))

    val in_last  = Input(Bool())
    val out_last = Output(Bool())

    val in_valid  = Input(Bool())
    val out_valid = Output(Bool())
  })

  // Instantiate MAC unit
  val mac_unit = Module(new MacUnit)

  // Input signals
  val a     = io.in_a
  val b     = io.in_b
  val d     = io.in_d
  val prop  = io.in_control.propagate
  // val shift = io.in_control.shift // Removed
  val id    = io.in_id
  val last  = io.in_last
  val valid = io.in_valid

  // Accumulation register

  val weight = Reg(UInt(7.W))
  // Pass-through signals
  io.out_a                 := a
  io.out_control.propagate := prop
  io.out_id                := id
  io.out_last              := last
  io.out_valid             := valid

  // MAC unit connections
  mac_unit.io.in_a := a
  mac_unit.io.in_b := d

  // Propagation control constant
  val PROPAGATE = 1.U(1.W)

  // Default assignment
  io.out_b         := b
  mac_unit.io.in_c := b
  // Weight Stationary mode
  when(prop === PROPAGATE) {
    when(valid) {
      weight := d
    }
    io.out_d         := d
    mac_unit.io.in_b := d
    mac_unit.io.in_c := b
    mac_unit.io.in_a := a
    io.out_b         := mac_unit.io.out_d
    io.out_a         := a
  }.otherwise {
    // Computation mode: output c2, use c1 as weight for computation
    io.out_a         := a
    io.out_d         := DontCare
    mac_unit.io.in_b := weight
    mac_unit.io.in_c := b
    mac_unit.io.in_a := a
    io.out_b         := mac_unit.io.out_d
  }

  // Do not update register when invalid
  when(!valid) {
    weight := weight
  }
}

class BBFP_PE_Array2x2 extends Module {

  val io = IO(new Bundle {
    // Row input activation (horizontal propagation)
    val in_a       = Input(Vec(2, UInt(7.W)))
    // Column input weight (vertical propagation)
    val in_d       = Input(Vec(2, UInt(7.W)))
    // Column input partial sum (vertical propagation)
    val in_b       = Input(Vec(2, UInt(32.W)))
    // Column input control signal (follows weight vertical propagation)
    val in_control = Input(Vec(2, new PEControl()))
    val in_id      = Input(Vec(2, UInt(1.W)))
    val in_last    = Input(Vec(2, Bool()))
    val in_valid   = Input(Vec(2, Bool()))

    // Output
    // Bottom row output partial sum
    val out_b = Output(Vec(2, UInt(32.W)))
    // Right column output activation
    val out_a = Output(Vec(2, UInt(7.W)))
    // Bottom row output weight
    val out_d = Output(Vec(2, UInt(7.W)))
  })

  // 2x2 PE array
  val pes = Seq.fill(2, 2)(Module(new BBFP_PE_WS()))

  // Registers connecting between PEs
  // Activation horizontal propagation register (row direction, left->right)
  val reg_a_h = Seq.fill(2)(Reg(UInt(7.W))) // pes(i)(0) -> pes(i)(1)

  // Weight vertical propagation register (column direction, top->bottom)
  val reg_d_v = Seq.fill(2)(Reg(UInt(7.W))) // pes(0)(j) -> pes(1)(j)

  // Partial sum vertical propagation register (column direction, top->bottom)
  val reg_b_v = Seq.fill(2)(Reg(UInt(32.W))) // pes(0)(j) -> pes(1)(j)

  // Control signal vertical propagation register (column direction, top->bottom)
  val reg_ctrl_v  = Seq.fill(2)(Wire(new PEControl))
  val reg_id_v    = Seq.fill(2)(Wire(UInt(1.W)))
  val reg_last_v  = Seq.fill(2)(Wire(Bool()))
  val reg_valid_v = Seq.fill(2)(Wire(Bool()))

  // ================ PE(0,0) ================
  // Row 0 activation input
  pes(0)(0).io.in_a       := io.in_a(0)
  // Column 0 weight input
  pes(0)(0).io.in_d       := io.in_d(0)
  // Column 0 partial sum input
  pes(0)(0).io.in_b       := io.in_b(0)
  // Column 0 control signal input
  pes(0)(0).io.in_control := io.in_control(0)
  pes(0)(0).io.in_id      := io.in_id(0)
  pes(0)(0).io.in_last    := io.in_last(0)
  pes(0)(0).io.in_valid   := io.in_valid(0)

  // ================ PE(0,1) ================
  // Activation propagates horizontally from PE(0,0) (through register)
  reg_a_h(0)              := pes(0)(0).io.out_a
  pes(0)(1).io.in_a       := reg_a_h(0)
  // Weight, partial sum, control signal input from external column 1
  pes(0)(1).io.in_d       := io.in_d(1)
  pes(0)(1).io.in_b       := io.in_b(1)
  pes(0)(1).io.in_control := io.in_control(1)
  pes(0)(1).io.in_id      := io.in_id(1)
  pes(0)(1).io.in_last    := io.in_last(1)
  pes(0)(1).io.in_valid   := io.in_valid(1)

  // ================ PE(1,0) ================
  // Activation input from external row 1
  pes(1)(0).io.in_a := io.in_a(1)
  // Weight, partial sum, control signal propagate vertically from PE(0,0) (through register)
  reg_d_v(0)        := pes(0)(0).io.out_d
  reg_b_v(0)        := pes(0)(0).io.out_b
  reg_ctrl_v(0)     := pes(0)(0).io.out_control
  reg_id_v(0)       := pes(0)(0).io.out_id
  reg_last_v(0)     := pes(0)(0).io.out_last
  reg_valid_v(0)    := pes(0)(0).io.out_valid

  pes(1)(0).io.in_d       := reg_d_v(0)
  pes(1)(0).io.in_b       := reg_b_v(0)
  pes(1)(0).io.in_control := reg_ctrl_v(0)
  pes(1)(0).io.in_id      := reg_id_v(0)
  pes(1)(0).io.in_last    := reg_last_v(0)
  pes(1)(0).io.in_valid   := reg_valid_v(0)

  // ================ PE(1,1) ================
  // Activation propagates horizontally from PE(1,0) (through register)
  reg_a_h(1)        := pes(1)(0).io.out_a
  pes(1)(1).io.in_a := reg_a_h(1)
  // Weight, partial sum, control signal propagate vertically from PE(0,1) (through register)
  reg_d_v(1)        := pes(0)(1).io.out_d
  reg_b_v(1)        := pes(0)(1).io.out_b
  reg_ctrl_v(1)     := pes(0)(1).io.out_control
  reg_id_v(1)       := pes(0)(1).io.out_id
  reg_last_v(1)     := pes(0)(1).io.out_last
  reg_valid_v(1)    := pes(0)(1).io.out_valid

  pes(1)(1).io.in_d       := reg_d_v(1)
  pes(1)(1).io.in_b       := reg_b_v(1)
  pes(1)(1).io.in_control := reg_ctrl_v(1)
  pes(1)(1).io.in_id      := reg_id_v(1)
  pes(1)(1).io.in_last    := reg_last_v(1)
  pes(1)(1).io.in_valid   := reg_valid_v(1)

  // ================ Output connections ================
  // Add registers for outputs
  val out_b_reg = Reg(Vec(2, UInt(32.W)))
  val out_a_reg = Reg(Vec(2, UInt(7.W)))
  val out_d_reg = Reg(Vec(2, UInt(7.W)))

  out_b_reg(0) := pes(1)(0).io.out_b
  out_b_reg(1) := pes(1)(1).io.out_b
  out_a_reg(0) := pes(0)(1).io.out_a
  out_a_reg(1) := pes(1)(1).io.out_a
  out_d_reg(0) := pes(1)(0).io.out_d
  out_d_reg(1) := pes(1)(1).io.out_d

  io.out_b := out_b_reg
  io.out_a := out_a_reg
  io.out_d := out_d_reg
}

class BBFP_PE_Array16x16 extends Module {

  val io = IO(new Bundle {
    // Row input activation (horizontal propagation) - 16 rows
    val in_a       = Input(Vec(16, UInt(7.W)))
    // Column input weight (vertical propagation) - 16 columns
    val in_d       = Input(Vec(16, UInt(7.W)))
    // Column input partial sum (vertical propagation) - 16 columns
    val in_b       = Input(Vec(16, UInt(32.W)))
    // Column input control signal (follows weight vertical propagation) - 16 columns
    val in_control = Input(Vec(16, new PEControl()))
    val in_id      = Input(Vec(16, UInt(1.W)))
    val in_last    = Input(Vec(16, Bool()))
    val in_valid   = Input(Vec(16, Bool()))

    // Output
    // Bottom row output partial sum
    val out_b = Output(Vec(16, UInt(32.W)))
    // Right column output activation
    val out_a = Output(Vec(16, UInt(7.W)))
    // Bottom row output weight
    val out_d = Output(Vec(16, UInt(7.W)))
  })

  // 16x16 PE array
  val pes = Seq.fill(16, 16)(Module(new BBFP_PE_WS()))

  // Registers connecting between PEs
  // Activation horizontal propagation register (row direction, left->right)
  val reg_a_h = Seq.fill(16, 15)(Reg(UInt(7.W))) // pes(i)(j) -> pes(i)(j+1)

  // Weight vertical propagation register (column direction, top->bottom)
  val reg_d_v = Seq.fill(15, 16)(Reg(UInt(7.W))) // pes(i)(j) -> pes(i+1)(j)

  // Partial sum vertical propagation register (column direction, top->bottom)
  val reg_b_v = Seq.fill(15, 16)(Reg(UInt(32.W))) // pes(i)(j) -> pes(i+1)(j)

  // Control signal vertical propagation register (column direction, top->bottom)
  val reg_ctrl_v  = Seq.fill(15, 16)(Wire(new PEControl))
  val reg_id_v    = Seq.fill(15, 16)(Wire(UInt(1.W)))
  val reg_last_v  = Seq.fill(15, 16)(Wire(Bool()))
  val reg_valid_v = Seq.fill(15, 16)(Wire(Bool()))

  // ================ PE array connections ================
  for (i <- 0 until 16) {
    for (j <- 0 until 16) {
      // Activation input connection (horizontal propagation)
      if (j == 0) {
        // First column: input from external
        pes(i)(j).io.in_a := io.in_a(i)
      } else {
        // Other columns: propagate from left PE through register
        reg_a_h(i)(j - 1) := pes(i)(j - 1).io.out_a
        pes(i)(j).io.in_a := reg_a_h(i)(j - 1)
      }

      // Weight input connection (vertical propagation)
      if (i == 0) {
        // First row: input from external
        pes(i)(j).io.in_d := io.in_d(j)
      } else {
        // Other rows: propagate from top PE through register
        reg_d_v(i - 1)(j) := pes(i - 1)(j).io.out_d
        pes(i)(j).io.in_d := reg_d_v(i - 1)(j)
      }

      // Partial sum input connection (vertical propagation)
      if (i == 0) {
        // First row: input from external
        pes(i)(j).io.in_b := io.in_b(j)
      } else {
        // Other rows: propagate from top PE through register
        reg_b_v(i - 1)(j) := pes(i - 1)(j).io.out_b
        pes(i)(j).io.in_b := reg_b_v(i - 1)(j)
      }

      // Control signal input connection (vertical propagation)
      if (i == 0) {
        // First row: input from external
        pes(i)(j).io.in_control := io.in_control(j)
        pes(i)(j).io.in_id      := io.in_id(j)
        pes(i)(j).io.in_last    := io.in_last(j)
        pes(i)(j).io.in_valid   := io.in_valid(j)
      } else {
        // Other rows: propagate from top PE through register
        reg_ctrl_v(i - 1)(j)  := pes(i - 1)(j).io.out_control
        reg_id_v(i - 1)(j)    := pes(i - 1)(j).io.out_id
        reg_last_v(i - 1)(j)  := pes(i - 1)(j).io.out_last
        reg_valid_v(i - 1)(j) := pes(i - 1)(j).io.out_valid

        pes(i)(j).io.in_control := reg_ctrl_v(i - 1)(j)
        pes(i)(j).io.in_id      := reg_id_v(i - 1)(j)
        pes(i)(j).io.in_last    := reg_last_v(i - 1)(j)
        pes(i)(j).io.in_valid   := reg_valid_v(i - 1)(j)
      }
    }
  }

  // ================ Output connections ================
  // Add registers for outputs
  val out_b_reg = Reg(Vec(16, UInt(32.W)))
  val out_a_reg = Reg(Vec(16, UInt(7.W)))
  val out_d_reg = Reg(Vec(16, UInt(7.W)))

  // Bottom row output partial sum (all columns of row 15)
  for (j <- 0 until 16) {
    out_b_reg(j) := pes(15)(j).io.out_b
  }

  // Right column output activation (row 15 of all rows)
  for (i <- 0 until 16) {
    out_a_reg(i) := pes(i)(15).io.out_a
  }

  // Bottom row output weight (all columns of row 15)
  for (j <- 0 until 16) {
    out_d_reg(j) := pes(15)(j).io.out_d
  }

  io.out_b := out_b_reg
  io.out_a := out_a_reg
  io.out_d := out_d_reg
}
