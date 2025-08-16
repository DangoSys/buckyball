// package framework.builtin.frontend.rs

// import chisel3._
// import chisel3.util._
// import examples.BuckyBallConfigs.CustomBuckyBallConfig
// import framework.builtin.util.Util._

// class IssueQueue(queue_entries: Int = 4)(implicit bbconfig: CustomBuckyBallConfig) extends Module {
//   val cmd_t = new BuckyBallCmd
//   val rob_id_width = log2Up(bbconfig.rob_entries)
  
//   val io = IO(new Bundle {
//     val rob_issue_i = Flipped(Decoupled(new BuckyBallCmd))  // 来自ROB的指令
//     val issue_o = new RSISSInterface(cmd_t, rob_id_width)   // 发给执行单元
//   })

//   // 根据指令类型分发到不同的队列
//   val ld_queue = Module(new Queue(new ReservationStationIssue(cmd_t, rob_id_width), queue_entries))
//   val st_queue = Module(new Queue(new ReservationStationIssue(cmd_t, rob_id_width), queue_entries))
//   val ex_queue = Module(new Queue(new ReservationStationIssue(cmd_t, rob_id_width), queue_entries))
  
//   // 创建发射请求
//   val issue_req = Wire(new ReservationStationIssue(cmd_t, rob_id_width))
//   issue_req.cmd := io.rob_issue_i.bits
//   issue_req.rob_id := io.rob_issue_i.bits.rob_id
  
//   // 根据指令类型决定发送到哪个队列
//   val is_load = io.rob_issue_i.bits.cmd_type === 1.U
//   val is_store = io.rob_issue_i.bits.cmd_type === 2.U
//   val is_ex = io.rob_issue_i.bits.cmd_type === 3.U
  
//   // 连接到对应的队列
//   ld_queue.io.enq.valid := io.rob_issue_i.valid && is_load
//   ld_queue.io.enq.bits := issue_req
  
//   st_queue.io.enq.valid := io.rob_issue_i.valid && is_store
//   st_queue.io.enq.bits := issue_req
  
//   ex_queue.io.enq.valid := io.rob_issue_i.valid && is_ex
//   ex_queue.io.enq.bits := issue_req
  
//   // ROB ready 当至少有一个队列可以接收时
//   io.rob_issue_i.ready := (is_load && ld_queue.io.enq.ready) || 
//                           (is_store && st_queue.io.enq.ready) || 
//                           (is_ex && ex_queue.io.enq.ready)
  
//   // 将队列的输出直接连接到issue接口
//   io.issue_o.ld <> ld_queue.io.deq
//   io.issue_o.st <> st_queue.io.deq
//   io.issue_o.ex <> ex_queue.io.deq
// }