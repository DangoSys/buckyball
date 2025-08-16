// package framework.builtin.frontend.rs

// import chisel3._
// import chisel3.util._
// import examples.BuckyBallConfigs.CustomBuckyBallConfig
// import framework.builtin.frontend.PostDecodeCmd
// import org.chipsalliance.cde.config.Parameters

// class BuckyBallCmd(implicit bbconfig: CustomBuckyBallConfig, p: Parameters) extends Bundle {
//   val post_decode_cmd = new PostDecodeCmd
//   val rob_id          = UInt(log2Up(bbconfig.rob_entries).W)
//   val cmd_type        = UInt(3.W) // 01: Load, 10: Store, 11: Ex, 100: Fence
// }

// class NextROBIdCounter(implicit bbconfig: CustomBuckyBallConfig, p: Parameters) extends Module {
//   val rob_entries = bbconfig.rob_entries
  
//   val io = IO(new Bundle {
//     val post_decode_cmd_i = Flipped(Decoupled(new PostDecodeCmd))
//     val post_index_cmd_o = new Bundle {
//       val cmd          = Decoupled(new BuckyBallCmd)
//       val new_head_ptr = Output(UInt(log2Up(rob_entries).W))
//     }
//     val rob_cmt_i = Flipped(Decoupled(UInt(log2Up(rob_entries).W))) // 传入robid，但没实际用
//   })

//   val head_ptr = RegInit(0.U(log2Up(rob_entries).W))
//   val tail_ptr = RegInit(0.U(log2Up(rob_entries).W))

//   // 状态管理 - 尾指针的下一个位置等于头指针就是满了
//   val full = ((tail_ptr + 1.U) % rob_entries.U) === head_ptr
//   val empty = tail_ptr === head_ptr

//   // id->ROBCounter
//   io.post_decode_cmd_i.ready := !full && io.post_index_cmd_o.cmd.ready
//   // ROBCounter
//   when(io.post_decode_cmd_i.fire && !io.post_decode_cmd_i.bits.is_fence) {
//     tail_ptr               := (tail_ptr + 1.U) % rob_entries.U
//   }
//   // ROBCounter->ROB
//   when(io.post_decode_cmd_i.fire) {
//     io.post_index_cmd_o.cmd.valid                := true.B
//     io.post_index_cmd_o.cmd.bits.rob_id          := tail_ptr
//     io.post_index_cmd_o.cmd.bits.cmd_type        := Mux(io.post_decode_cmd_i.bits.is_load, 1.U, 
//                                                      Mux(io.post_decode_cmd_i.bits.is_store, 2.U, 
//                                                        Mux(io.post_decode_cmd_i.bits.is_ex, 3.U, 
//                                                        Mux(io.post_decode_cmd_i.bits.is_fence, 4.U,0.U))))
//     io.post_index_cmd_o.cmd.bits.post_decode_cmd := io.post_decode_cmd_i.bits
//     io.post_index_cmd_o.new_head_ptr             := head_ptr
//   }.otherwise {
//     io.post_index_cmd_o.cmd.valid                              := false.B
//     io.post_index_cmd_o.cmd.bits.rob_id                        := 0.U
//     io.post_index_cmd_o.cmd.bits.cmd_type                      := 0.U
//     io.post_index_cmd_o.cmd.bits.post_decode_cmd.is_load       := false.B
//     io.post_index_cmd_o.cmd.bits.post_decode_cmd.is_store      := false.B
//     io.post_index_cmd_o.cmd.bits.post_decode_cmd.is_ex         := false.B
//     io.post_index_cmd_o.cmd.bits.post_decode_cmd.is_fence      := false.B
//     io.post_index_cmd_o.cmd.bits.post_decode_cmd.mem_addr      := 0.U
//     io.post_index_cmd_o.cmd.bits.post_decode_cmd.iter          := 0.U
//     io.post_index_cmd_o.cmd.bits.post_decode_cmd.rd_bank       := 0.U
//     io.post_index_cmd_o.cmd.bits.post_decode_cmd.rd_bank_addr  := 0.U
//     io.post_index_cmd_o.cmd.bits.post_decode_cmd.wr_bank       := 0.U
//     io.post_index_cmd_o.cmd.bits.post_decode_cmd.wr_bank_addr  := 0.U
//     io.post_index_cmd_o.cmd.bits.post_decode_cmd.is_acc        := false.B
//     io.post_index_cmd_o.cmd.bits.post_decode_cmd.op1_en        := false.B
//     io.post_index_cmd_o.cmd.bits.post_decode_cmd.op2_en        := false.B
//     io.post_index_cmd_o.cmd.bits.post_decode_cmd.wr_spad_en    := false.B
//     io.post_index_cmd_o.cmd.bits.post_decode_cmd.op1_from_spad := false.B
//     io.post_index_cmd_o.cmd.bits.post_decode_cmd.op2_from_spad := false.B
//     io.post_index_cmd_o.cmd.bits.post_decode_cmd.op1_bank      := 0.U
//     io.post_index_cmd_o.cmd.bits.post_decode_cmd.op1_bank_addr := 0.U
//     io.post_index_cmd_o.cmd.bits.post_decode_cmd.op2_bank      := 0.U
//     io.post_index_cmd_o.cmd.bits.post_decode_cmd.op2_bank_addr := 0.U
//     io.post_index_cmd_o.cmd.bits.post_decode_cmd.pid           := 0.U
//     io.post_index_cmd_o.cmd.bits.post_decode_cmd.pstart        := false.B
//     io.post_index_cmd_o.cmd.bits.post_decode_cmd.pend          := false.B
//     io.post_index_cmd_o.new_head_ptr                           := 0.U
//     io.post_index_cmd_o.cmd.bits.post_decode_cmd.is_vec        := false.B
//     io.post_index_cmd_o.cmd.bits.post_decode_cmd.is_bbfp       := false.B
//     io.post_index_cmd_o.cmd.bits.post_decode_cmd.is_matmul_ws  := false.B
//   }


//   // ROB->ROBCounter
//   io.rob_cmt_i.ready := !empty
//   when(io.rob_cmt_i.fire) {
//     head_ptr := (head_ptr + 1.U) % rob_entries.U
//   }
// }
