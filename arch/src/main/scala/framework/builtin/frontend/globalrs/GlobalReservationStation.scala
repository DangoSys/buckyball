package framework.builtin.frontend.globalrs

import chisel3._
import chisel3.util._
import chisel3.experimental._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import framework.builtin.frontend.PostGDCmd
import framework.rocket.RoCCResponseBB

// 全局ROB条目 - 只包含基本信息，不包含具体指令解码
class GlobalRobEntry(implicit b: CustomBuckyBallConfig, p: Parameters) extends Bundle {
  val cmd    = new PostGDCmd
  val rob_id = UInt(log2Up(b.rob_entries).W)
}

// 全局RS的发射接口
class GlobalRsIssue(implicit b: CustomBuckyBallConfig, p: Parameters) extends GlobalRobEntry

// 全局RS的完成接口
class GlobalRsComplete(implicit b: CustomBuckyBallConfig, p: Parameters) extends Bundle {
  val rob_id = UInt(log2Up(b.rob_entries).W)
}

// 不需要额外的接口Bundle，直接在IO里定义

// 全局保留站 - 在GlobalDecoder和各Domain之间
class GlobalReservationStation(implicit b: CustomBuckyBallConfig, p: Parameters) extends Module {
  val io = IO(new Bundle {
    // GlobalDecoder -> 全局RS
    val global_decode_cmd_i = Flipped(new DecoupledIO(new PostGDCmd))

    // 全局RS -> BallDomain (单通道)
    val ball_issue_o = Decoupled(new GlobalRsIssue)

    // 全局RS -> MemDomain (单通道)
    val mem_issue_o = Decoupled(new GlobalRsIssue)

    // BallDomain -> 全局RS (单通道)
    val ball_complete_i = Flipped(Decoupled(new GlobalRsComplete))

    // MemDomain -> 全局RS (单通道)
    val mem_complete_i = Flipped(Decoupled(new GlobalRsComplete))

    // RoCC响应
    val rs_rocc_o = new Bundle {
      val resp  = new DecoupledIO(new RoCCResponseBB()(p))
      val busy  = Output(Bool())
    }
  })

  val rob = Module(new GlobalROB)

// -----------------------------------------------------------------------------
// Fence处理
// -----------------------------------------------------------------------------
  val fenceActive = RegInit(false.B)
  val isFenceCmd = io.global_decode_cmd_i.valid && io.global_decode_cmd_i.bits.is_fence  // 不能用fire，会形成环路
  val robEmpty = rob.io.empty

  // Fence状态机：只有当fence指令被接受（fire）时才激活
  when (io.global_decode_cmd_i.fire && isFenceCmd && !fenceActive) {
    fenceActive := true.B
  }
  when (fenceActive && robEmpty) {
    fenceActive := false.B
  }

// -----------------------------------------------------------------------------
// 入站 - 指令分配（Fence指令不进ROB）
// -----------------------------------------------------------------------------
  // 过滤掉fence指令
  rob.io.alloc.valid := io.global_decode_cmd_i.valid && !isFenceCmd
  rob.io.alloc.bits  := io.global_decode_cmd_i.bits

  // 反压逻辑：
  // - 普通指令：等待ROB ready
  // - Fence指令：等待ROB empty
  io.global_decode_cmd_i.ready := Mux(isFenceCmd, robEmpty, rob.io.alloc.ready)

// -----------------------------------------------------------------------------
// 出站 - 指令发射 (根据is_ball/is_mem分发到对应域)
// -----------------------------------------------------------------------------
  // Ball域发射
  io.ball_issue_o.valid := rob.io.issue.valid && rob.io.issue.bits.cmd.is_ball
  io.ball_issue_o.bits  := rob.io.issue.bits

  // Mem域发射
  io.mem_issue_o.valid := rob.io.issue.valid && rob.io.issue.bits.cmd.is_mem
  io.mem_issue_o.bits  := rob.io.issue.bits

  // 设置ROB的ready信号 - 只有目标域ready时才能发射
  rob.io.issue.ready :=
    (rob.io.issue.bits.cmd.is_ball && io.ball_issue_o.ready) ||
    (rob.io.issue.bits.cmd.is_mem  && io.mem_issue_o.ready)

// -----------------------------------------------------------------------------
// 完成信号处理
// -----------------------------------------------------------------------------
  val completeArb = Module(new Arbiter(UInt(log2Up(b.rob_entries).W), 2))

  // 连接Ball和Mem域的完成信号到仲裁器
  completeArb.io.in(0).valid := io.ball_complete_i.valid
  completeArb.io.in(0).bits  := io.ball_complete_i.bits.rob_id
  io.ball_complete_i.ready := completeArb.io.in(0).ready

  completeArb.io.in(1).valid := io.mem_complete_i.valid
  completeArb.io.in(1).bits  := io.mem_complete_i.bits.rob_id
  io.mem_complete_i.ready := completeArb.io.in(1).ready

  // 根据配置决定是否过滤完成信号
  if (b.rs_out_of_order_response) {
    // 乱序模式：接受所有完成信号，ROB内部乱序提交
    rob.io.complete <> completeArb.io.out
  } else {
    // 顺序模式：只接受rob_id == head_ptr的完成信号
    val isHeadComplete = completeArb.io.out.bits === rob.io.head_ptr
    rob.io.complete.valid := completeArb.io.out.valid && isHeadComplete
    rob.io.complete.bits  := completeArb.io.out.bits
    completeArb.io.out.ready := rob.io.complete.ready && isHeadComplete
  }

// -----------------------------------------------------------------------------
// 响应生成
// -----------------------------------------------------------------------------
  // 响应逻辑：
  // - 普通指令：立即响应
  // - Fence指令：等ROB空后响应
  io.rs_rocc_o.resp.valid := io.global_decode_cmd_i.fire &&
                             (!io.global_decode_cmd_i.bits.is_fence || robEmpty)
  io.rs_rocc_o.resp.bits  := DontCare
  io.rs_rocc_o.busy       := !rob.io.empty || fenceActive
}
