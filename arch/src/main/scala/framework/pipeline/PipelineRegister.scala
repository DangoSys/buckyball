// package framework.pipeline

// import chisel3._
// import chisel3.util._
// import chisel3.stage._

// // Sisyphus table entry
// // Sisyphus表条目
// class SisyphusTableEntry extends Bundle {
//   val id           = UInt(8.W)        // Sisyphus unique identifier
//   val operation    = UInt(8.W)        // Operation type
//   val pipelineId   = UInt(1.W)        // Which pipeline (0 or 1)
//   val stageId      = UInt(4.W)        // Which stage in the pipeline
//   val rankId       = UInt(4.W)        // Which rank in the stage
//   val iteration    = UInt(32.W)       // Iteration count for this Sisyphus
//   val status       = UInt(2.W)        // Status of this Sisyphus 1: idle, 2: ready, 3: running 0: not registered
// }

// class PipelineRegisterIO extends Bundle {
//   val register = Flipped(Decoupled(new SisyphusTableEntry))  
// }

// class PipelineRegister(val sisyphusNum: Int) extends Module {
//   val io = IO(new PipelineRegisterIO)
  
//   val sisyphusTable = Reg(Vec(sisyphusNum, new SisyphusTableEntry))
  
//   // Initialize table
//   for (i <- 0 until sisyphusNum) {
//     sisyphusTable(i).id         := i.U
//     sisyphusTable(i).operation  := 0.U
//     sisyphusTable(i).pipelineId := 0.U
//     sisyphusTable(i).stageId    := 0.U
//     sisyphusTable(i).rankId     := 0.U
//     sisyphusTable(i).iteration  := 0.U
//     sisyphusTable(i).status     := 0.U
//   }
  
//   // Sisyphus registration logic
//   io.register.ready := true.B
  
//   when(io.register.valid) {
//     sisyphusTable(io.register.bits.id) := io.register.bits
//   }
  
  
//   // Query functions - all queries implemented as functions
//   // 查询函数 - 所有查询都用函数实现
  
//   /**
//    * Get Sisyphus table entry by ID
//    * 根据ID获取Sisyphus表条目
//    */
//   def getSisyphus(sisyphusId: UInt): SisyphusTable = {
//     sisyphusTable(sisyphusId)
//   }
  
//   /**
//    * Get Sisyphus operation type
//    * 获取Sisyphus操作类型
//    */
//   def getOperation(sisyphusId: UInt): UInt = {
//     sisyphusTable(sisyphusId).operation
//   }
  
//   /**
//    * Get Sisyphus position
//    * 获取Sisyphus位置
//    */
//   def getPosition(sisyphusId: UInt): (UInt, UInt) = {
//     (sisyphusTable(sisyphusId).pipelineId, sisyphusTable(sisyphusId).stageId)
//   }
  
//   /**
//    * Get Sisyphus iteration count
//    * 获取Sisyphus迭代次数
//    */
//   def getIteration(sisyphusId: UInt): UInt = {
//     sisyphusTable(sisyphusId).iteration
//   }
  
//   /**
//    * Get all Sisyphus IDs at specific pipeline stage
//    * 获取特定流水线阶段的所有Sisyphus ID
//    */
//   def getSisyphusAtStage(pipelineId: UInt, stageId: UInt): Vec[UInt] = {
//     val matches = sisyphusTable.map(s => 
//       s.isRegistered && 
//       (s.pipelineId === pipelineId) && 
//       (s.stageId === stageId)
//     )
    
//     val idList = Wire(Vec(maxSisyphus, UInt(8.W)))
//     for (i <- 0 until maxSisyphus) {
//       when(matches(i)) {
//         idList(i) := i.U
//       }.otherwise {
//         idList(i) := 0.U
//       }
//     }
//     idList
//   }
  
//   /**
//    * Get all Sisyphus IDs at specific pipeline stage and rank
//    * 获取特定流水线阶段和排序的Sisyphus ID
//    */
//   def getSisyphusAtStageAndRank(pipelineId: UInt, stageId: UInt, rankId: UInt): Vec[UInt] = {
//     val matches = sisyphusTable.map(s => 
//       s.isRegistered && 
//       (s.pipelineId === pipelineId) && 
//       (s.stageId === stageId) && 
//       (s.rankId === rankId)
//     )
    
//     val idList = Wire(Vec(sisyphusNum, UInt(8.W)))
//     for (i <- 0 until sisyphusNum) {
//       when(matches(i)) {
//         idList(i) := i.U
//       }.otherwise {
//         idList(i) := 0.U
//       }
//     }
//     idList
//   }
  
// }