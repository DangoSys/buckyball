// package framework.pipeline

// import chisel3._
// import chisel3.util._
// import chisel3.stage._

// // Pipeline manager interface
// // 流水线管理器接口
// class PipelineManagerIO(numStages: Int) extends Bundle {
//   // Global pipeline control
//   // 全局流水线控制
//   val enable = Input(Bool())
//   val reset_pipeline = Input(Bool())
  
//   // Configuration interface
//   // 配置接口
//   val config = Flipped(Decoupled(new PipelineStageConfig))
//   val num_stages = Input(UInt(8.W))
  
//   // Data input/output
//   // 数据输入输出
//   val data_in = Flipped(Decoupled(Vec(16, UInt(32.W))))
//   val data_out = Decoupled(Vec(16, UInt(32.W)))
  
//   // Individual stage controls (for external Sisyphus modules)
//   // 各个阶段的控制（用于外部Sisyphus模块）
//   val stage_commands = Vec(numStages, new SisyphusCmd)
//   val stage_controls = Vec(numStages, Flipped(new SisyphusCtl))
//   val stage_data = Vec(numStages, new SisyphusData)
  
//   // Pipeline status
//   // 流水线状态
//   val pipeline_ready = Output(Bool())
//   val pipeline_busy = Output(Bool())
//   val stage_status = Output(Vec(numStages, new Bundle {
//     val gate_open = Bool()
//     val stage_busy = Bool()
//     val stage_error = Bool()
//   }))
// }

// class PipelineManager(numStages: Int = 4) extends Module {
//   val io = IO(new PipelineManagerIO(numStages))
  
//   // Instantiate pipeline register
//   // 实例化流水线寄存器
//   val pipelineRegister = Module(new PipelineRegister(numStages))
  
//   // Instantiate pipeline supervisors for each stage
//   // 为每个阶段实例化流水线监督器
//   val supervisors = Seq.fill(numStages)(Module(new PipelineSupervisor))
  
//   // Connect pipeline register
//   // 连接流水线寄存器
//   pipelineRegister.io.config <> io.config
//   pipelineRegister.io.numStages := io.num_stages
//   pipelineRegister.io.reset_pipeline := io.reset_pipeline
  
//   // Configure each supervisor
//   // 配置每个监督器
//   for (i <- 0 until numStages) {
//     val supervisor = supervisors(i)
    
//     // Basic stage configuration
//     // 基本阶段配置
//     supervisor.io.stage_id := i.U
//     supervisor.io.stage_enable := io.enable && pipelineRegister.io.pipeline_ready
//     supervisor.io.stage_reset := io.reset_pipeline
    
//     // Connect to external Sisyphus modules
//     // 连接到外部Sisyphus模块
//     supervisor.io.sisyphus_cmd <> io.stage_commands(i)
//     supervisor.io.sisyphus_ctl <> io.stage_controls(i)
//     supervisor.io.sisyphus_data <> io.stage_data(i)
    
//     // Connect data flow between stages
//     // 连接各阶段间的数据流
//     if (i == 0) {
//       // First stage connects to pipeline input
//       // 第一阶段连接到流水线输入
//       supervisor.io.upstream_data <> io.data_in
//     } else {
//       // Other stages connect to previous stage output
//       // 其他阶段连接到前一阶段的输出
//       supervisor.io.upstream_data <> supervisors(i-1).io.downstream_data
//       supervisors(i-1).io.downstream_ready := supervisor.io.upstream_ready
//     }
    
//     if (i == numStages - 1) {
//       // Last stage connects to pipeline output
//       // 最后阶段连接到流水线输出
//       supervisor.io.downstream_data <> io.data_out
//       supervisor.io.downstream_ready := io.data_out.ready
//     }
    
//     // Collect status information
//     // 收集状态信息
//     io.stage_status(i).gate_open := supervisor.io.gate_open
//     io.stage_status(i).stage_busy := supervisor.io.stage_busy
//     io.stage_status(i).stage_error := supervisor.io.stage_error
//   }
  
//   // Pipeline-level status signals
//   // 流水线级别的状态信号
//   io.pipeline_ready := pipelineRegister.io.pipeline_ready && io.enable
//   io.pipeline_busy := supervisors.map(_.io.stage_busy).reduce(_ || _)
  
//   // Debug information
//   // 调试信息
//   when(io.pipeline_ready && !RegNext(io.pipeline_ready)) {
//     printf(p"[PipelineManager] Pipeline configured and ready with ${io.num_stages} stages\n")
//   }
  
//   when(io.pipeline_busy && !RegNext(io.pipeline_busy)) {
//     printf(p"[PipelineManager] Pipeline processing started\n")
//   }
  
//   when(!io.pipeline_busy && RegNext(io.pipeline_busy)) {
//     printf(p"[PipelineManager] Pipeline processing completed\n")
//   }
// } 