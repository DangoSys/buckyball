package framework.pipeline

import chisel3._
import chisel3.util._
import chisel3.stage._

// Pipeline supervisor interface for connecting to Sisyphus modules
// 用于连接Sisyphus模块的流水线监督器接口
class PipelineSupervisorIO extends Bundle {
  // Connection to the supervised Sisyphus control signals only
  // 只连接到被监督的Sisyphus控制信号
  val sisyphus_ctl = Flipped(new SisyphusCtl)
  
  // Generic data interface - can connect to any Decoupled data stream
  // 通用数据接口 - 可以连接到任何Decoupled数据流
  val data_in = Flipped(Decoupled(Vec(16, UInt(32.W))))
  val data_out = Decoupled(Vec(16, UInt(32.W)))
  
  // Stage control and configuration
  // 阶段控制和配置
  val stage_id = Input(UInt(8.W))
  val stage_enable = Input(Bool())
  val stage_reset = Input(Bool())
  
  // Status outputs
  // 状态输出
  val gate_open = Output(Bool())
  val stage_busy = Output(Bool())
}

class PipelineSupervisor extends Module {
  val io = IO(new PipelineSupervisorIO)
  
  // Simple gate control - no state machine needed
  // 简单的门控制 - 不需要状态机
  val gateOpen = RegInit(false.B)
  val prevArrive = RegInit(false.B)
  val prevFinish = RegInit(false.B)
  
  // Reset logic
  // 复位逻辑
  when(io.stage_reset) {
    gateOpen := false.B
    prevArrive := false.B
    prevFinish := false.B
  }
  
  // Detect rising edges of arrive and finish signals
  // 检测arrive和finish信号的上升沿
  val arriveRisingEdge = io.sisyphus_ctl.arrive && !prevArrive
  val finishRisingEdge = io.sisyphus_ctl.finish && !prevFinish
  
  prevArrive := io.sisyphus_ctl.arrive
  prevFinish := io.sisyphus_ctl.finish
  
  // Simple gate control logic
  // 简单的门控逻辑
  when(arriveRisingEdge && io.stage_enable) {
    gateOpen := true.B
  }.elsewhen(finishRisingEdge) {
    gateOpen := false.B
  }
  
  // Data flow control - gate controls the data passing
  // 数据流控制 - 门控制数据通过
  io.data_out.valid := io.data_in.valid && gateOpen
  io.data_out.bits := io.data_in.bits
  
  // Backpressure control - input ready only when output ready and gate open
  // 背压控制 - 只有当输出准备好且门开启时输入才准备好
  io.data_in.ready := io.data_out.ready && gateOpen
  
  // Status outputs
  // 状态输出
  io.gate_open := gateOpen
  io.stage_busy := io.sisyphus_ctl.start || io.sisyphus_ctl.arrive
}
