package framework.pipeline

import chisel3._
import chisel3.util._
import chisel3.stage._


class SisyphusCmd extends Bundle {
  val Operation = Flipped(Decoupled(UInt(8.W)))  // Operation command from external
  val Iteration = Input(UInt(32.W))               // Iteration count from external
}

class SisyphusCtl extends Bundle {
  val start  = Output(Bool())                  // Start processing signal (controlled by subclass)
  val arrive = Output(Bool())                 // Indicates processing has started (controlled by subclass)
  val finish = Output(Bool())                 // Indicates processing is complete (controlled by subclass)
}

class SisyphusData extends Bundle {
  val dataIn  = Flipped(Decoupled(Vec(16, UInt(32.W))))      // Input 16 numbers per cycle during processing
  val dataOut = Decoupled(Vec(16, UInt(32.W)))               // Output 16 numbers per cycle during processing
}

// Sisyphus抽象基类 - 提供流水线模块的基础框架
abstract class Sisyphus extends Module {
  val io = IO(new Bundle {
    // Command interface
    val cmd = new SisyphusCmd

    // Control signals
    val ctl = new SisyphusCtl
    
    // Data interface - 可以被子类覆盖以定义具体的输入输出
    val data = new SisyphusData
  })
  
  // States for the state machine
  object State extends ChiselEnum {
    val sIDLE, sPREPARE, sPROCESSING = Value
  }
  
  // State registers that can be accessed by subclasses
  // 可被子类访问的状态寄存器
  protected val state        = RegInit(State.sIDLE)
  protected val cycleCounter = RegInit(0.U(32.W))
  
  // Control signal wires that subclasses can drive
  // 子类可以驱动的控制信号线
  protected val startSignal  = WireDefault(false.B)
  protected val arriveSignal = WireDefault(false.B)
  protected val finishSignal = WireDefault(false.B)
  
  // Abstract methods that must be implemented by subclasses
  // 子类必须实现的抽象方法
  
  /**
   * Returns the number of cycles this module needs to process
   * 返回此模块需要处理的周期数
   */
  def getProcessingCycles(): UInt

  /**
   * Returns the operation code of this module
   * 返回此模块的操作码
   */
  def getOperation(): UInt

  /**
   * Returns the iteration of this module
   * 返回此模块arrive后的总迭代次数
   */
  def getIteration(): UInt

  /**
   * Process function called each cycle during processing state
   * Should set arrive and finish signals appropriately
   * Returns the 16 output values for current cycle
   * 每个处理周期调用的处理函数，应适当设置arrive和finish信号
   * 返回当前周期的16个输出值
   */
  def processData(cycle: UInt): Vec[UInt]
  
  /**
   * Optional initialization function called when starting processing
   * 可选的初始化函数，在开始处理时调用
   */
  def initProcessing(): Unit = {}
  
  /**
   * Optional cleanup function called when finishing processing
   * 可选的清理函数，在完成处理时调用
   */
  def finishProcessing(): Unit = {}
  
  // Get processing duration from subclass implementation
  val processingCycles = getProcessingCycles()
  
  // State machine logic
  switch(state) {
    is(State.sIDLE) {
      when(startSignal) {
        state := State.sPROCESSING
        cycleCounter := getIteration()
        initProcessing() // Call subclass initialization
      }
    }
    is(State.sPREPARE) {
      when(arriveSignal) {
        state := State.sPROCESSING
        cycleCounter := getIteration()
      }
    }
    is(State.sPROCESSING) {
      cycleCounter := cycleCounter - 1.U
      when(finishSignal) {
        state := State.sIDLE
        finishProcessing() // Call subclass cleanup
      }
    }
    
  }
  
  // Connect control signals
  io.ctl.start  := startSignal
  io.ctl.arrive := arriveSignal
  io.ctl.finish := finishSignal
  
  // Command interface ready signal
  // 命令接口就绪信号
  io.cmd.Operation.ready := state === State.sIDLE
  
  // Data output - call subclass processing function
  // 数据输出 - 调用子类处理函数
  io.data.dataOut.valid := state === State.sPROCESSING && arriveSignal
  io.data.dataOut.bits := Mux(state =/= State.sPROCESSING, processData(cycleCounter), VecInit(Seq.fill(16)(0.U(32.W))))
  
  // Input data ready signal (can be overridden by subclasses)
  // 输入数据就绪信号（可被子类覆盖）
  io.data.dataIn.ready := state === State.sPREPARE
}