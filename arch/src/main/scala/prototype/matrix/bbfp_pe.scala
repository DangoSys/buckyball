package prototype.matrix
import chisel3._
import chisel3.util._
import chisel3.stage._
import prototype.matrix._
// PE控制信号Bundle
class PEControl extends Bundle {
  val propagate = UInt(1.W)   // 传播控制
}

class MacUnit extends Module {
  val io = IO(new Bundle {
  val in_a  = Input(UInt(7.W))   // 无符号输入：[6]=sign, [5]=flag, [4:0]=value
  val in_b  = Input(UInt(7.W))   // 无符号输入：[6]=sign, [5]=flag, [4:0]=value
  val in_c  = Input(UInt(32.W))  // 无符号输入：[31]=sign, [30:0]=value
  val out_d = Output(UInt(32.W)) // 有符号输出
  })

  // 提取符号位
  val sign_a = io.in_a(6)
  val sign_b = io.in_b(6)
  val sign_c = io.in_c(31)

  // 提取flag位
  val flag_a = io.in_a(5)
  val flag_b = io.in_b(5)

  // 提取数值部分
  val value_a = io.in_a(4, 0)  // 5位数值
  val value_b = io.in_b(4, 0)  // 5位数值
  val value_c = io.in_c(30, 0) // 31位数值


  // val extended_a = value_a.asUInt().pad(7) // 扩展到 7 位
  // val extended_b = value_b.asUInt().pad(7) // 扩展到 7 位


  // 根据flag位决定是否左移
  val shifted_a = Mux(flag_a === 1.U, value_a << 2, value_a)
  val shifted_b = Mux(flag_b === 1.U, value_b << 2, value_b)


  // 将数值转换为有符号数，考虑符号位
  // 先扩展位宽避免溢出，然后根据符号位决定正负
  val a_signed = Mux(sign_a === 1.U,
  -(shifted_a.zext),
  shifted_a.zext
  ).asSInt

  val b_signed = Mux(sign_b === 1.U,
  -(shifted_b.zext),
  shifted_b.zext
  ).asSInt

  val c_signed = Mux(sign_c === 1.U,
  -(value_c.zext),
  value_c.zext
  ).asSInt

  // 执行MAC运算：a * b + c
  val result = a_signed * b_signed + c_signed

  // 输出结果
  io.out_d := result.asUInt
}


// BBFP PE单元（仅支持Weight Stationary）
class BBFP_PE_WS(max_simultaneous_matmuls: Int = 16) extends Module {
  val io = IO(new Bundle {
  // 数据输入输出
  val in_a = Input(UInt(7.W))  // 输入激活值
  val in_b = Input(UInt(32.W))   // 输入部分和
  val in_d = Input(UInt(7.W))   // 输入权重
  val out_a = Output(UInt(7.W))  // 输出激活值
  val out_b = Output(UInt(32.W)) // 输出部分和
  val out_d = Output(UInt(7.W)) // 输出权重
  // 控制信号
  val in_control = Input(new PEControl())
  val out_control = Output(new PEControl())

  // ID和有效信号
  val in_id = Input(UInt(log2Up(max_simultaneous_matmuls).W))
  val out_id = Output(UInt(log2Up(max_simultaneous_matmuls).W))

  val in_last = Input(Bool())
  val out_last = Output(Bool())

  val in_valid = Input(Bool())
  val out_valid = Output(Bool())
  })

  // MAC单元实例化
  val mac_unit = Module(new MacUnit)

  // 输入信号
  val a = io.in_a
  val b = io.in_b
  val d = io.in_d
  val prop = io.in_control.propagate
  // val shift = io.in_control.shift // 已删除
  val id = io.in_id
  val last = io.in_last
  val valid = io.in_valid

  // 累加寄存器

  val weight = Reg(UInt(7.W))
  // 直通信号
  io.out_a := a
  io.out_control.propagate := prop
  io.out_id := id
  io.out_last := last
  io.out_valid := valid

  // MAC单元连接
  mac_unit.io.in_a := a
  mac_unit.io.in_b := d

  // 传播控制常量
  val PROPAGATE = 1.U(1.W)

  // 默认赋值
  io.out_b := b
  mac_unit.io.in_c := b
  // Weight Stationary 模式
  when(prop === PROPAGATE) {
  when(valid) {
    weight := d
  }
  io.out_d := d
  mac_unit.io.in_b := d
  mac_unit.io.in_c := b
  mac_unit.io.in_a := a
  io.out_b := mac_unit.io.out_d
  io.out_a := a
  }.otherwise {
  // 计算模式：输出c2，用c1作为权重计算
  io.out_a := a
  io.out_d := DontCare
  mac_unit.io.in_b := weight
  mac_unit.io.in_c := b
  mac_unit.io.in_a := a
  io.out_b := mac_unit.io.out_d
  }

  // 当无效时，不更新寄存器
  when(!valid) {
  weight := weight
  }
}



class BBFP_PE_Array2x2 extends Module {
  val io = IO(new Bundle {
  // 行输入激活（横向传播）
  val in_a = Input(Vec(2, UInt(7.W)))
  // 列输入权重（竖向传播）
  val in_d = Input(Vec(2, UInt(7.W)))
  // 列输入部分和（竖向传播）
  val in_b = Input(Vec(2, UInt(32.W)))
  // 列输入控制信号（跟随权重竖向传播）
  val in_control = Input(Vec(2, new PEControl()))
  val in_id = Input(Vec(2, UInt(1.W)))
  val in_last = Input(Vec(2, Bool()))
  val in_valid = Input(Vec(2, Bool()))

  // 输出
  val out_b = Output(Vec(2, UInt(32.W)))  // 底行输出部分和
  val out_a = Output(Vec(2, UInt(7.W)))   // 右列输出激活
  val out_d = Output(Vec(2, UInt(7.W)))   // 底行输出权重
  })

  // 2x2 PE阵列
  val pes = Seq.fill(2, 2)(Module(new BBFP_PE_WS()))

  // PE间连接的寄存器
  // 激活横向传播寄存器 (行方向，左->右)
  val reg_a_h = Seq.fill(2)(Reg(UInt(7.W)))  // pes(i)(0) -> pes(i)(1)

  // 权重竖向传播寄存器 (列方向，上->下)
  val reg_d_v = Seq.fill(2)(Reg(UInt(7.W)))  // pes(0)(j) -> pes(1)(j)

  // 部分和竖向传播寄存器 (列方向，上->下)
  val reg_b_v = Seq.fill(2)(Reg(UInt(32.W))) // pes(0)(j) -> pes(1)(j)

  // 控制信号竖向传播寄存器 (列方向，上->下)
  val reg_ctrl_v   = Seq.fill(2)(Wire(new PEControl))
  val reg_id_v   = Seq.fill(2)(Wire(UInt(1.W)))
  val reg_last_v   = Seq.fill(2)(Wire(Bool()))
  val reg_valid_v  = Seq.fill(2)(Wire(Bool()))

  // ================ PE(0,0) ================
  pes(0)(0).io.in_a     := io.in_a(0)    // 行0激活输入
  pes(0)(0).io.in_d     := io.in_d(0)    // 列0权重输入
  pes(0)(0).io.in_b     := io.in_b(0)    // 列0部分和输入
  pes(0)(0).io.in_control := io.in_control(0)  // 列0控制信号输入
  pes(0)(0).io.in_id    := io.in_id(0)
  pes(0)(0).io.in_last  := io.in_last(0)
  pes(0)(0).io.in_valid   := io.in_valid(0)

  // ================ PE(0,1) ================
  // 激活从PE(0,0)横向传播过来（通过寄存器）
  reg_a_h(0) := pes(0)(0).io.out_a
  pes(0)(1).io.in_a     := reg_a_h(0)
  // 权重、部分和、控制信号从外部列1输入
  pes(0)(1).io.in_d     := io.in_d(1)
  pes(0)(1).io.in_b     := io.in_b(1)
  pes(0)(1).io.in_control := io.in_control(1)
  pes(0)(1).io.in_id    := io.in_id(1)
  pes(0)(1).io.in_last  := io.in_last(1)
  pes(0)(1).io.in_valid   := io.in_valid(1)

  // ================ PE(1,0) ================
  // 激活从外部行1输入
  pes(1)(0).io.in_a     := io.in_a(1)
  // 权重、部分和、控制信号从PE(0,0)竖向传播过来（通过寄存器）
  reg_d_v(0)  := pes(0)(0).io.out_d
  reg_b_v(0)  := pes(0)(0).io.out_b
  reg_ctrl_v(0) := pes(0)(0).io.out_control
  reg_id_v(0)   := pes(0)(0).io.out_id
  reg_last_v(0) := pes(0)(0).io.out_last
  reg_valid_v(0):= pes(0)(0).io.out_valid

  pes(1)(0).io.in_d     := reg_d_v(0)
  pes(1)(0).io.in_b     := reg_b_v(0)
  pes(1)(0).io.in_control := reg_ctrl_v(0)
  pes(1)(0).io.in_id    := reg_id_v(0)
  pes(1)(0).io.in_last  := reg_last_v(0)
  pes(1)(0).io.in_valid   := reg_valid_v(0)

  // ================ PE(1,1) ================
  // 激活从PE(1,0)横向传播过来（通过寄存器）
  reg_a_h(1) := pes(1)(0).io.out_a
  pes(1)(1).io.in_a     := reg_a_h(1)
  // 权重、部分和、控制信号从PE(0,1)竖向传播过来（通过寄存器）
  reg_d_v(1)  := pes(0)(1).io.out_d
  reg_b_v(1)  := pes(0)(1).io.out_b
  reg_ctrl_v(1) := pes(0)(1).io.out_control
  reg_id_v(1)   := pes(0)(1).io.out_id
  reg_last_v(1) := pes(0)(1).io.out_last
  reg_valid_v(1):= pes(0)(1).io.out_valid

  pes(1)(1).io.in_d     := reg_d_v(1)
  pes(1)(1).io.in_b     := reg_b_v(1)
  pes(1)(1).io.in_control := reg_ctrl_v(1)
  pes(1)(1).io.in_id    := reg_id_v(1)
  pes(1)(1).io.in_last  := reg_last_v(1)
  pes(1)(1).io.in_valid   := reg_valid_v(1)

  // ================ 输出连接 ================
  // 为输出添加寄存器
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
  // 行输入激活（横向传播）- 16行
  val in_a = Input(Vec(16, UInt(7.W)))
  // 列输入权重（竖向传播）- 16列
  val in_d = Input(Vec(16, UInt(7.W)))
  // 列输入部分和（竖向传播）- 16列
  val in_b = Input(Vec(16, UInt(32.W)))
  // 列输入控制信号（跟随权重竖向传播）- 16列
  val in_control = Input(Vec(16, new PEControl()))
  val in_id = Input(Vec(16, UInt(1.W)))
  val in_last = Input(Vec(16, Bool()))
  val in_valid = Input(Vec(16, Bool()))

  // 输出
  val out_b = Output(Vec(16, UInt(32.W)))  // 底行输出部分和
  val out_a = Output(Vec(16, UInt(7.W)))   // 右列输出激活
  val out_d = Output(Vec(16, UInt(7.W)))   // 底行输出权重
  })

  // 16x16 PE阵列
  val pes = Seq.fill(16, 16)(Module(new BBFP_PE_WS()))

  // PE间连接的寄存器
  // 激活横向传播寄存器 (行方向，左->右)
  val reg_a_h = Seq.fill(16, 15)(Reg(UInt(7.W)))  // pes(i)(j) -> pes(i)(j+1)

  // 权重竖向传播寄存器 (列方向，上->下)
  val reg_d_v = Seq.fill(15, 16)(Reg(UInt(7.W)))  // pes(i)(j) -> pes(i+1)(j)

  // 部分和竖向传播寄存器 (列方向，上->下)
  val reg_b_v = Seq.fill(15, 16)(Reg(UInt(32.W))) // pes(i)(j) -> pes(i+1)(j)

  // 控制信号竖向传播寄存器 (列方向，上->下)
  val reg_ctrl_v   = Seq.fill(15, 16)(Wire(new PEControl))
  val reg_id_v   = Seq.fill(15, 16)(Wire(UInt(1.W)))
  val reg_last_v   = Seq.fill(15, 16)(Wire(Bool()))
  val reg_valid_v  = Seq.fill(15, 16)(Wire(Bool()))

  // ================ PE阵列连接 ================
  for (i <- 0 until 16) {
  for (j <- 0 until 16) {
    // 激活输入连接（横向传播）
    if (j == 0) {
    // 第一列：从外部输入
    pes(i)(j).io.in_a := io.in_a(i)
    } else {
    // 其他列：从左侧PE通过寄存器传播
    reg_a_h(i)(j-1) := pes(i)(j-1).io.out_a
    pes(i)(j).io.in_a := reg_a_h(i)(j-1)
    }

    // 权重输入连接（竖向传播）
    if (i == 0) {
    // 第一行：从外部输入
    pes(i)(j).io.in_d := io.in_d(j)
    } else {
    // 其他行：从上方PE通过寄存器传播
    reg_d_v(i-1)(j) := pes(i-1)(j).io.out_d
    pes(i)(j).io.in_d := reg_d_v(i-1)(j)
    }

    // 部分和输入连接（竖向传播）
    if (i == 0) {
    // 第一行：从外部输入
    pes(i)(j).io.in_b := io.in_b(j)
    } else {
    // 其他行：从上方PE通过寄存器传播
    reg_b_v(i-1)(j) := pes(i-1)(j).io.out_b
    pes(i)(j).io.in_b := reg_b_v(i-1)(j)
    }

    // 控制信号输入连接（竖向传播）
    if (i == 0) {
    // 第一行：从外部输入
    pes(i)(j).io.in_control := io.in_control(j)
    pes(i)(j).io.in_id    := io.in_id(j)
    pes(i)(j).io.in_last  := io.in_last(j)
    pes(i)(j).io.in_valid   := io.in_valid(j)
    } else {
    // 其他行：从上方PE通过寄存器传播
    reg_ctrl_v(i-1)(j) := pes(i-1)(j).io.out_control
    reg_id_v(i-1)(j)   := pes(i-1)(j).io.out_id
    reg_last_v(i-1)(j) := pes(i-1)(j).io.out_last
    reg_valid_v(i-1)(j):= pes(i-1)(j).io.out_valid

    pes(i)(j).io.in_control := reg_ctrl_v(i-1)(j)
    pes(i)(j).io.in_id    := reg_id_v(i-1)(j)
    pes(i)(j).io.in_last  := reg_last_v(i-1)(j)
    pes(i)(j).io.in_valid   := reg_valid_v(i-1)(j)
    }
  }
  }

  // ================ 输出连接 ================
  // 为输出添加寄存器
  val out_b_reg = Reg(Vec(16, UInt(32.W)))
  val out_a_reg = Reg(Vec(16, UInt(7.W)))
  val out_d_reg = Reg(Vec(16, UInt(7.W)))

  // 底行输出部分和（第15行的所有列）
  for (j <- 0 until 16) {
  out_b_reg(j) := pes(15)(j).io.out_b
  }

  // 右列输出激活（所有行的第15列）
  for (i <- 0 until 16) {
  out_a_reg(i) := pes(i)(15).io.out_a
  }

  // 底行输出权重（第15行的所有列）
  for (j <- 0 until 16) {
  out_d_reg(j) := pes(15)(j).io.out_d
  }

  io.out_b := out_b_reg
  io.out_a := out_a_reg
  io.out_d := out_d_reg
}
