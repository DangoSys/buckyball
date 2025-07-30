// package framework.pipeline

// import chisel3._
// import chisel3.util._

// // Vector addition module - performs element-wise addition of two 16-element vectors
// // 向量加法模块 - 执行两个16元素向量的逐元素加法
// class VectorAdd extends Module {
//   val io = IO(new Bundle {
//     val dataInA = Input(Vec(16, UInt(32.W)))     // First input vector
//     val dataInB = Input(Vec(16, UInt(32.W)))     // Second input vector
//     val dataOut = Output(Vec(16, UInt(32.W)))    // Output vector (A + B)
//     val enable  = Input(Bool())                  // Enable signal for computation
//   })
  
//   // Perform element-wise addition when enabled
//   // 当使能时执行逐元素加法
//   for (i <- 0 until 16) {
//     io.dataOut(i) := Mux(io.enable, io.dataInA(i) + io.dataInB(i), 0.U)
//   }
// } 

// class VectorAddSisyphusData extends SisyphusData {
//   val dataInA = Flipped(Decoupled(Vec(16, UInt(32.W))))
//   val dataInB = Flipped(Decoupled(Vec(16, UInt(32.W))))
//   val dataOut = Decoupled(Vec(16, UInt(32.W)))
// }

// class VectorAddSisyphus extends Sisyphus {
  
//   override val io = IO(new Bundle {
//     val cmd = new SisyphusCmd
//     val ctl = new SisyphusCtl
//     val data = new VectorAddSisyphusData
//   })
  
//   // Instantiate the vector addition module
//   // 实例化向量加法模块
//   val vectorAddModule = Module(new VectorAdd)
  
//   def getProcessingCycles(): UInt = cycleCounter
//   def getOperation(): UInt = io.cmd.Operation.bits
//   def getIteration(): UInt = io.cmd.Iteration
  
//   def processData(cycle: UInt): Vec[UInt] = {
//     startSignal  := io.cmd.Operation.fire
//     arriveSignal := io.data.dataOut.fire
//     finishSignal := io.data.dataOut.fire && (cycleCounter === 0.U)
    
//     // Connect inputs to the vector addition module
//     // 连接输入到向量加法模块
//     vectorAddModule.io.dataInA := Mux(io.data.dataInA.valid, io.data.dataInA.bits, VecInit(Seq.fill(16)(0.U(32.W))))
//     vectorAddModule.io.dataInB := Mux(io.data.dataInB.valid, io.data.dataInB.bits, VecInit(Seq.fill(16)(0.U(32.W))))
//     vectorAddModule.io.enable := io.data.dataInA.valid && io.data.dataInB.valid
    
//     // Return the output from the vector addition module
//     // 返回向量加法模块的输出
//     vectorAddModule.io.dataOut
//   }
  
//   // Input ready signals - both inputs must be valid for processing  
//   // 输入就绪信号 - 两个输入都必须有效才能处理
//   io.data.dataInA.ready := (state === State.sPROCESSING) && io.data.dataInB.valid
//   io.data.dataInB.ready := (state === State.sPROCESSING) && io.data.dataInA.valid
  
// } 