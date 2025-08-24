// package framework.builtin.frontend.rs

// import chisel3._
// import chisel3.util._
// import org.chipsalliance.cde.config.Parameters
// import examples.BuckyBallConfigs.CustomBuckyBallConfig
// import framework.builtin.util.Util._
// import freechips.rocketchip.tile._  



// object RoBState extends ChiselEnum {
//   val sInvalid  = Value   // 无效状态
//   val sWaiting  = Value   // 等待分发
//   val sIssued   = Value   // 已发射到执行单元
//   val sComplete = Value   // 执行完成，等待提交
// }

// class RoBEntry(implicit b: CustomBuckyBallConfig) extends Bundle {
//   val state    = RoBState()  // 声明类型而不是赋予默认值
//   val cmd      = new BuckyBallCmd
//   val cmd_type = UInt(3.W)
//   val ready    = Bool()  // 前置指令是否发射完成
  
//   def is_ready = ready && (state === RoBState.sWaiting)
//   def can_commit = state === RoBState.sComplete
// }

// // Buckyball的ROB只有一个队列所有指令都按照FIFO顺序执行
// // 顺序执行，Load/Store/Ex每次最多只会各自发射一个
// // 为保证没有读写冲突，现在版本只支持Load/Ex并行和Store/Ex并行，不支持同时执行Load/Store
// // 如果RoB内顺序为Ex -> Ex -> Store，则这条Store指令不会发射，
// // 等后续流水线指令设计完成才可以发射整条流水线
// // 最终也只能顺序提交
// class ReorderBuffer(implicit b: CustomBuckyBallConfig, p: Parameters) extends Module {
//   val rob_entries = b.rob_entries
//   val rob_id_width = log2Up(b.rob_entries)
//   val cmd_t = new BuckyBallCmd
  
//   val io = IO(new Bundle {
//     val post_indexed_cmd_i = new Bundle {
//       val cmd          = Flipped(Decoupled(new BuckyBallCmd))
//       val new_head_ptr = Input(UInt(log2Up(b.rob_entries).W))
//     }
//     val issue_o        = Decoupled(new BuckyBallCmd)  // 按顺序发射到ISSQueue
//     val commit_i       = Flipped(Valid(UInt(rob_id_width.W)))
//     // to top
//     val rob_cmt_o     = new Bundle {      
//       val resp        = Decoupled(new RoCCResponse()(p))
//       val busy        = Output(Bool())  // 是否有指令在ROB中等待提交
//     }
//     // to ROBCounter
//     val rob_robcnt_o  = Decoupled(UInt(log2Up(b.rob_entries).W))
//   })

//   // ROB条目数组
//   val RobEntries = Reg(Vec(rob_entries, new RoBEntry))
//   val fence_waiting = RegInit(false.B)  // 用于标志等待Fence指令完成
//   val rs_timer = RegInit(0.U(16.W))  // 用于跟踪RS的计时器
  
//   // 初始化ROB entries为无效状态
//   for (i <- 0 until rob_entries) {
//     when(reset.asBool) {
//       RobEntries(i).state := RoBState.sInvalid
//     }
//   }
  
// // -----------------------------------------------------------------------------
// // 入队
// // -----------------------------------------------------------------------------
//   io.post_indexed_cmd_i.cmd.ready := !fence_waiting
//   val cmd_i        = io.post_indexed_cmd_i.cmd.bits
//   val rob_id       = io.post_indexed_cmd_i.cmd.bits.rob_id
//   val cmd_type     = io.post_indexed_cmd_i.cmd.bits.cmd_type
//   val head_ptr     = io.post_indexed_cmd_i.new_head_ptr

//   when(io.post_indexed_cmd_i.cmd.fire && !fence_waiting) {
//     assert(RobEntries(rob_id).state === RoBState.sInvalid || 
//         RobEntries(rob_id).state === RoBState.sWaiting || 
//         RobEntries(rob_id).state === RoBState.sIssued, "Inserting to non-empty ROB entry")
//     when(cmd_type === 4.U) { // Fence指令
//       fence_waiting := RobEntries.map((entry: RoBEntry) => (entry.state === RoBState.sWaiting) || (entry.state === RoBState.sIssued)).reduce(_ || _)
//     }.otherwise {
//       RobEntries(rob_id).state    := RoBState.sWaiting
//       RobEntries(rob_id).cmd      := cmd_i
//       RobEntries(rob_id).cmd_type := cmd_type
//       RobEntries(rob_id).ready    := true.B
//     }
//   }
//   when(fence_waiting){
//    fence_waiting := RobEntries.map((entry: RoBEntry) => (entry.state === RoBState.sWaiting) || (entry.state === RoBState.sIssued)).reduce(_ || _)
//   }
//   when(RobEntries.map((entry: RoBEntry) => (entry.state === RoBState.sWaiting) || (entry.state === RoBState.sIssued)).reduce(_ || _)){
//     rs_timer := rs_timer + 1.U
//   } .otherwise {
//     rs_timer := 0.U
//   }
//   assert(rs_timer < 30000.U, "RS timer exceeded 30000 cycles without completion")
// // -----------------------------------------------------------------------------
// // 发射：按顺序发射到ISSQueue，考虑Load/Store互斥约束
// // -----------------------------------------------------------------------------
//   val issue_ptr = RegInit(0.U(log2Up(rob_entries).W))
  
//   // 跟踪已发射但未完成的指令类型
//   val load_in_flight = RegInit(false.B)
//   val store_in_flight = RegInit(false.B)
//   val ex_in_flight = RegInit(false.B)  // 跟踪Ex指令是否在执行中
  
//   // 跟踪Ex指令发射延迟
//   val ex_delay_counter = RegInit(0.U(10.W))  // 8-bit counter for 100 cycles
//   val last_issued_was_ex = RegInit(false.B)  // 跟踪上一次发射的是否为Ex指令
  
//   // 当前要发射的指令类型
//   val current_cmd_type = RobEntries(issue_ptr).cmd_type
//   val is_load = current_cmd_type === 1.U
//   val is_store = current_cmd_type === 2.U
//   val is_ball = current_cmd_type === 3.U
  
//   // 检查Load/Store互斥约束
//   val load_blocked = is_load && store_in_flight  // Load被正在执行的Store阻塞
//   val store_blocked = is_store && load_in_flight // Store被正在执行的Load阻塞
  
//   // 检查Ex指令延迟约束
//   val ex_delay_blocked = is_ball && last_issued_was_ex && (ex_delay_counter < 150.U)
  
//   val basic_can_issue = RobEntries(issue_ptr).state === RoBState.sWaiting && RobEntries(issue_ptr).ready
//   val can_issue = basic_can_issue && !load_blocked && !store_blocked && !ex_delay_blocked
  
//   io.issue_o.valid := can_issue
//   io.issue_o.bits := RobEntries(issue_ptr).cmd
  
//   when(io.issue_o.fire) {
//     RobEntries(issue_ptr).state := RoBState.sIssued
//     issue_ptr := (issue_ptr + 1.U) % rob_entries.U
    
//     // 更新in_flight标志
//     when(is_load) {
//       load_in_flight := true.B
//     }
//     when(is_store) {
//       store_in_flight := true.B
//     }
    
//     // 更新Ex指令延迟跟踪
//     when(is_ball) {
//       last_issued_was_ex := true.B
//       ex_delay_counter := 0.U  // 重置计数器
//     }.otherwise {
//       last_issued_was_ex := false.B
//       ex_delay_counter := 0.U  // 非Ex指令时也重置计数器
//     }
//   }
  
//   // 更新Ex指令延迟计数器
//   when(last_issued_was_ex && (ex_delay_counter < 150.U)) {
//     ex_delay_counter := ex_delay_counter + 1.U
//   }
  
// // -----------------------------------------------------------------------------
// // 提交
// // -----------------------------------------------------------------------------
//   // 出队
//   when(io.commit_i.valid) {
//     RobEntries(io.commit_i.bits).state := RoBState.sComplete
    
//     // 清除in_flight标志
//     val completed_cmd_type = RobEntries(io.commit_i.bits).cmd_type
//     when(completed_cmd_type === 1.U) { // Load完成
//       load_in_flight := false.B
//     }
//     when(completed_cmd_type === 2.U) { // Store完成
//       store_in_flight := false.B
//     }
//     when(completed_cmd_type === 3.U) { // Ex完成
//       ex_in_flight := false.B
//     }
//   }

//   // to ROBCounter
//   io.rob_robcnt_o.valid       := io.commit_i.valid
//   io.rob_robcnt_o.bits        := io.commit_i.bits
  
//   // 清理已完成的ROB entry
//   // 当ROBCounter接收到commit信号后，head_ptr会推进，此时可以清理对应的entry
//   when(io.rob_robcnt_o.fire) {
//     RobEntries(io.rob_robcnt_o.bits).state := RoBState.sInvalid
//   }
  
//   // to top
//   io.rob_cmt_o.resp.valid     := io.commit_i.valid
//   io.rob_cmt_o.resp.bits.rd   := 0.U
//   io.rob_cmt_o.resp.bits.data := 0.U

//   io.rob_cmt_o.busy           := fence_waiting && RobEntries.map(_.state =/= RoBState.sInvalid).reduce(_ || _)
// } 