package framework.balldomain.prototype.systolicarray

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.balldomain.rs.{BallRsComplete, BallRsIssue}
import framework.top.GlobalConfig

object SystolicArrayConst {
  // Ctrl、Load、EX、Store 共享的固定阵列参数：一个 tile 对应 16x16 的子矩阵。
  val Tile              = 16
  // A/B 操作数按 int8 传输；C 的累加结果按 int32 传输。
  val OpElemBits        = 8
  val AccElemBits       = 32
  val OpRowBits         = Tile * OpElemBits
  val ResultRowBits     = Tile * AccElemBits
  // Store 将一行 16 个 int32 结果拆成最多 4 个 128-bit bank 写 beat。
  val StoreWritePorts   = 4
  val StorePortElemCount = Tile / StoreWritePorts
  // WS 模式一次保留一组 B 权重，供最多 8 个 M tile 复用。
  val WsReuseTiles       = 8
}

object SystolicCtrlLoadReqKind {
  // OS：为当前 tile 同时读取 A 与 B，EX 使用 K-tile 链完成累加。
  val READ_AB       = 0.U(2.W)
  // WS：B 已驻留在 PE 中，只读取当前 M tile 的 A。
  val READ_A_ONLY   = 1.U(2.W)
  // WS：读取 A，同时用 B 刷新 PE 权重；B 只发送当前 K tile 的真实行数。
  val READ_A_B_PE   = 2.U(2.W)
  // WS：当前 A 仍复用 PE 权重，同时预取下一套 B 到 B buffer。
  val READ_A_B_BUF  = 3.U(2.W)
}

object SystolicKTileKind {
  // K 维只有一个 tile，无需与其他 K tile 串联累加。
  val DIRECT = 0.U(2.W)
  // 多个 K tile 时，分别标记累加链的起始、中间与结束 tile。
  val FIRST  = 1.U(2.W)
  val MIDDLE = 2.U(2.W)
  val LAST   = 3.U(2.W)
}

/** Ctrl 发给 Load 的单个 tile 读请求及其计算元数据。 */
class SystolicCtrlLoadReq(b: GlobalConfig) extends Bundle {
  // 决定读取 A/B 的组合方式，以及 EX 如何解释本请求。
  val req_kind     = UInt(2.W)
  // 描述当前 tile 在同一输出 tile 的 K 维累加链中的位置。
  val k_tile_kind  = UInt(2.W)
  // WS 模式的累加上下文槽；OS 不使用该字段，固定发送 0。
  val acc_slot     = UInt(log2Ceil(SystolicArrayConst.WsReuseTiles).W)
  // 边缘 tile 的真实 M/N/K 尺寸，Load/EX 用它们完成补零和结果裁剪。
  val valid_m      = UInt(5.W)
  val valid_n      = UInt(5.W)
  val valid_k      = UInt(5.W)
  // B buffer 的有效 N/K；普通请求与 valid_n/valid_k 相同。
  val b_valid_n    = UInt(5.W)
  val b_valid_k    = UInt(5.W)
  // 当前 A tile 计算时应选用的 PE 权重代际。
  val weight_generation = Bool()
  // A、B 各自的 bank、逻辑 group 与 group 内起始行地址。
  val op1_bank     = UInt(log2Up(b.memDomain.bankNum).W)
  val op1_group    = UInt(log2Up(b.memDomain.bankNum).W)
  val op1_row_base = UInt(log2Up(b.memDomain.bankEntries).W)
  val op2_bank     = UInt(log2Up(b.memDomain.bankNum).W)
  val op2_group    = UInt(log2Up(b.memDomain.bankNum).W)
  val op2_row_base = UInt(log2Up(b.memDomain.bankEntries).W)
}

/** EX 产出的一整行 C 结果；实际写 bank 前由 Store 拆分为若干 beat。 */
class SystolicResultRow extends Bundle {
  val data = UInt(SystolicArrayConst.ResultRowBits.W)
}

/** Ctrl 对一次 Store 行请求的地址与有效元素数响应。 */
class SystolicStoreCtrlResp(b: GlobalConfig) extends Bundle {
  // 本行真实列数（1..16）；Store 据此产生对应数量的写 beat。
  val row_valid_elems = UInt(5.W)
  // 这行结果所属的前端任务；Store 将其保留到实际 bank 写完成。
  val rob_id           = UInt(log2Up(b.frontend.rob_entries).W)
  val wr_bank         = UInt(log2Up(b.memDomain.bankNum).W)
  val wr_group_base   = UInt(log2Up(b.memDomain.bankNum).W)
  val wr_row_addr     = UInt(log2Up(b.memDomain.bankEntries).W)
}

/** Store 发给 bank 的单个写请求格式。Ctrl 不直接生成该请求，仅定义共享类型。 */
class SystolicStoreWriteReq(b: GlobalConfig) extends Bundle {
  val rob_id        = UInt(log2Up(b.frontend.rob_entries).W)
  val wr_bank       = UInt(log2Up(b.memDomain.bankNum).W)
  val wr_group_base = UInt(log2Up(b.memDomain.bankNum).W)
  val wr_row_addr   = UInt(log2Up(b.memDomain.bankEntries).W)
  val valid_elems   = UInt(5.W)
  val data          = UInt(SystolicArrayConst.ResultRowBits.W)
}

/**
  * 从一条发射命令中锁存的任务描述。
  *
  * 命令可以在 Ctrl 执行期间继续进入 issueQ；这里的寄存器保证当前任务的
  * 维度、bank 地址和 ROB 信息在整个执行周期保持不变。
  */
class SystolicTask(b: GlobalConfig) extends Bundle {
  // mode=false 为 OS，mode=true 为 WS。
  val mode       = Bool()
  val m          = UInt(12.W)
  val n          = UInt(12.W)
  val k          = UInt(12.W)
  val op1_bank   = UInt(log2Up(b.memDomain.bankNum).W)
  val op2_bank   = UInt(log2Up(b.memDomain.bankNum).W)
  val wr_bank    = UInt(log2Up(b.memDomain.bankNum).W)
  val op1_base   = UInt(log2Up(b.memDomain.bankEntries).W)
  val op2_base   = UInt(log2Up(b.memDomain.bankEntries).W)
  val wr_base    = UInt(log2Up(b.memDomain.bankEntries).W)
  val rob_id     = UInt(log2Up(b.frontend.rob_entries).W)
  val is_sub     = Bool()
  val sub_rob_id = UInt(log2Up(b.frontend.sub_rob_depth * 4).W)
}

class SystolicRetireMeta(b: GlobalConfig) extends Bundle {
  val mode       = Bool()
  val m          = UInt(12.W)
  val n          = UInt(12.W)
  val rob_id     = UInt(log2Up(b.frontend.rob_entries).W)
  val is_sub     = Bool()
  val sub_rob_id = UInt(log2Up(b.frontend.sub_rob_depth * 4).W)
}

class SystolicStoreMeta(b: GlobalConfig) extends Bundle {
  val mode    = Bool()
  val m       = UInt(12.W)
  val n       = UInt(12.W)
  val wr_bank = UInt(log2Up(b.memDomain.bankNum).W)
  val wr_base = UInt(log2Up(b.memDomain.bankEntries).W)
  val rob_id  = UInt(log2Up(b.frontend.rob_entries).W)
}

