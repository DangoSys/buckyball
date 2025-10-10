package prototype.nagisa.gelu

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyBallConfigs.CustomBuckyBallConfig

class GeluEXUnit(implicit b: CustomBuckyBallConfig, p: Parameters) extends Module {
  val io = IO(new Bundle {
    val ctrl_ex_i = Flipped(Decoupled(new CtrlExReq))
    val ld_ex_i   = Flipped(Decoupled(new LdExReq))
    val ex_st_o   = Decoupled(new ExStReq)
  })

  val idle :: busy :: Nil = Enum(2)
  val state = RegInit(idle)
  val iter_reg = RegInit(0.U(10.W))
  val iter_cnt = RegInit(0.U(10.W))

  // 默认值
  io.ctrl_ex_i.ready := false.B
  io.ex_st_o.valid   := false.B
  io.ex_st_o.bits.data   := VecInit(Seq.fill(b.veclane)(0.U(b.accType.getWidth.W)))
  io.ex_st_o.bits.iter   := 0.U
  io.ex_st_o.bits.is_acc := false.B

  // 控制信号
  io.ctrl_ex_i.ready := state === idle
  when(io.ctrl_ex_i.fire) {
    state := busy
    // iter信息将从Load单元获得，这里只需要准备接收数据
    iter_cnt := 0.U
  }

  // 接收load数据并计算
  io.ld_ex_i.ready := state === busy

  // 迭代计数 - 使用从Load单元接收到的iter信息
  when(io.ld_ex_i.fire) {
    iter_reg := io.ld_ex_i.bits.iter  // 从Load单元获得迭代次数
    iter_cnt := iter_cnt + 1.U
    // 注意：不要在接收数据时立即转为idle，要等所有数据处理完
  }

  // GELU计算流水线
  // 使用简化的GELU近似: GELU(x) ≈ x * sigmoid(1.702 * x)
  // sigmoid(x) ≈ 0.5 + 0.5 * tanh(x/2)
  // 进一步简化为分段线性近似

  val gelu_result = Wire(Vec(b.veclane, UInt(b.accType.getWidth.W)))

  for (i <- 0 until b.veclane) {
    val x = io.ld_ex_i.bits.data(i).asSInt

    // 简化的GELU近似实现
    // 对于INT8输入：GELU(x) ≈ x 当 x > 0
    //                        ≈ 0 当 x < -threshold
    //                        ≈ x/2 当 -threshold < x < 0

    val threshold = 3.S  // 阈值
    val result = Wire(SInt(b.accType.getWidth.W))

    when(x >= threshold) {
      // x > threshold: GELU(x) ≈ x
      result := x
    }.elsewhen(x <= -threshold) {
      // x < -threshold: GELU(x) ≈ 0
      result := 0.S
    }.elsewhen(x >= 0.S) {
      // 0 <= x < threshold: GELU(x) ≈ 0.5*x + 0.5*x = x (简化)
      result := x
    }.otherwise {
      // -threshold < x < 0: GELU(x) ≈ 0.5*x (简化线性近似)
      result := (x >> 1)
    }

    gelu_result(i) := result.asUInt
  }

  // 流水线寄存器
  val result_valid_reg = RegInit(false.B)
  val result_data_reg  = Reg(Vec(b.veclane, UInt(b.accType.getWidth.W)))
  val result_iter_reg  = RegInit(0.U(10.W))
  val result_is_acc_reg = RegInit(false.B)

  when(io.ld_ex_i.fire) {
    result_valid_reg  := true.B
    result_data_reg   := gelu_result
    result_iter_reg   := io.ld_ex_i.bits.iter  // 捕获iter信息
    result_is_acc_reg := io.ld_ex_i.bits.is_acc
  }.elsewhen(io.ex_st_o.ready) {
    result_valid_reg := false.B
  }

  // 状态转换逻辑 - 当处理完所有数据后转为idle
  when(iter_cnt === iter_reg && iter_reg =/= 0.U && !result_valid_reg) {
    state := idle
    iter_cnt := 0.U
  }

  io.ex_st_o.valid       := result_valid_reg
  io.ex_st_o.bits.data   := result_data_reg
  io.ex_st_o.bits.iter   := result_iter_reg
  io.ex_st_o.bits.is_acc := result_is_acc_reg
}
