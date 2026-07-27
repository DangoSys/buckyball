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

@instantiable
class SystolicArrayCtrl(val b: GlobalConfig) extends Module {
  /**
    * 脉动阵列控制器。
    *
    * Ctrl 将一条 MxNxK 矩阵乘命令切分为 16x16x16 tile：一侧向 Load 发出
    * 数据/元数据请求，另一侧在 Store 请求写入时给出 C 行的目标地址。只有
    * 全部 tile 已发射且全部结果行写完后，才向前端返回该命令的完成响应。
    */
  private val tile       = SystolicArrayConst.Tile
  private val abRows     = b.memDomain.bankEntries
  private val bankWidth  = log2Up(b.memDomain.bankNum)
  private val groupWidth = log2Up(b.memDomain.bankNum)
  private val addrWidth  = log2Up(b.memDomain.bankEntries)
  private val countWidth = 32

  require(
    b.memDomain.bankEntries % tile == 0,
    "SystolicArrayCtrl expects bankEntries to be an integer number of 16-row A/B tiles"
  )

  @public
  val io = IO(new Bundle {
    // 前端命令入口与完成出口。cmdReq 可排队，cmdResp_o 仅对应当前 task。
    val cmdReq    = Flipped(Decoupled(new BallRsIssue(b)))
    val cmdResp_o = Decoupled(new BallRsComplete(b))

    // Load 接收一个完整 tile 的读地址、有效范围和 EX 元数据。
    val ctrl_ld_o = Decoupled(new SystolicCtrlLoadReq(b))

    // Ctrl 主动按结果顺序发送写回描述符；Store 以 ready 反馈其两项地址 FIFO 空位。
    val store_ctrl_resp_o = Decoupled(new SystolicStoreCtrlResp(b))
    val store_done_i      = Input(Bool())

    val busy_o = Output(Bool())
  })

  // 将可能宽于或窄于目标字段的地址/计数截断或零扩展到目标宽度。
  private def fitTo(x: UInt, width: Int): UInt =
    if (x.getWidth >= width) x(width - 1, 0) else x.pad(width)

  // 维度按 16 向上取整，得到 M/N/K 方向的 tile 个数。
  private def ceilDiv16(x: UInt): UInt =
    ((x.pad(countWidth) +& (tile - 1).U) >> log2Ceil(tile)).pad(countWidth)

  // A/B 的每个 tile 固定占用 16 行。先计算相对起始行，再拆成 bank group 和
  // group 内地址；包含补零行的固定 tile 布局仍可跨越 group 边界。
  private def abGroup(base: UInt, rowOffset: UInt): UInt = {
    val linear = base.pad(countWidth) + rowOffset.pad(countWidth)
    fitTo(linear / abRows.U, groupWidth)
  }

  private def abRow(base: UInt, rowOffset: UInt): UInt = {
    val linear = base.pad(countWidth) + rowOffset.pad(countWidth)
    fitTo(linear % abRows.U, addrWidth)
  }

  // C 的四个 128-bit beat 条带化到连续四个 group。一个线性 C block 对应一行
  // 16 个 int32：groupBase+0..3 分别保存列 0..3、4..7、8..11、12..15；同一
  // group quartet 的地址相同。每个物理 group 写满后，下一 quartet 从 group+4
  // 开始，因此四路写会命中四块独立的单端口 SRAM。
  private def cGroupBase(base: UInt, blockCount: UInt): UInt = {
    val linearBlock = base.pad(countWidth) + blockCount.pad(countWidth)
    val groupPage   = linearBlock / b.memDomain.bankEntries.U
    fitTo(groupPage << log2Ceil(SystolicArrayConst.StoreWritePorts), groupWidth)
  }

  private def cRowAddr(base: UInt, blockCount: UInt): UInt = {
    val linearBlock = base.pad(countWidth) + blockCount.pad(countWidth)
    fitTo(linearBlock % b.memDomain.bankEntries.U, addrWidth)
  }

  // 返回边缘 tile 的真实长度；完整 tile 为 16，最后一个 tile 可能更短。
  private def validTileExtent(dim: UInt, tileIndex: UInt): UInt = {
    val remaining = dim.pad(countWidth) - (tileIndex << log2Ceil(tile))
    Mux(remaining >= tile.U, tile.U(5.W), fitTo(remaining, 5))
  }

  // 命令队列接收前端任务；退休队列按同一顺序保留每个任务的 Store 地址状态。
  // 两者分离后，新任务的 Load tile 不必等待旧任务写回完成。
  val issueQ = Module(new Queue(new BallRsIssue(b), entries = 4))
  issueQ.io.enq.valid := io.cmdReq.valid
  issueQ.io.enq.bits  := io.cmdReq.bits
  io.cmdReq.ready     := issueQ.io.enq.ready

  val retireQ = Module(new Queue(new SystolicRetireMeta(b), entries = 4))
  val storeMetaQ = Module(new Queue(new SystolicStoreMeta(b), entries = 4))
  val completeQ = Module(new Queue(new BallRsComplete(b), entries = 4))
  val completionReservations = RegInit(0.U(log2Ceil(4 + 1).W))

  // task 与其 tile 游标只描述当前发射任务。它的最后一个 tile 被 Load 接收后，
  // 可在下一个周期直接装入 issueQ 的后继任务，而旧任务仍停留在 retireQ 队首。
  val issueActive = RegInit(false.B)
  val task = RegInit(0.U.asTypeOf(new SystolicTask(b)))

  // 当前发射任务的三个逻辑维度 tile 总数和 (M,N,K) 发射坐标。
  val mTiles = RegInit(0.U(countWidth.W))
  val nTiles = RegInit(0.U(countWidth.W))
  val kTiles = RegInit(0.U(countWidth.W))

  val mt = RegInit(0.U(countWidth.W))
  val nt = RegInit(0.U(countWidth.W))
  val kt = RegInit(0.U(countWidth.W))
  // WS 一批最多复用 8 个 M tile；wsBatchBase 是该批第一个 M tile 的下标。
  val wsBatchBase = RegInit(0.U(countWidth.W))

  // WS 下 wsLoadPeB=true 表示本次请求要以 B 更新 PE 权重；false 时仅读 A。
  val wsLoadPeB = RegInit(false.B)
  val wsWeightGeneration = RegInit(false.B)
  val wsPrefetchedWeights = RegInit(false.B)

  // 完成游标只随 store_done_i 前进，用于判断 retireQ 队首任务何时真正写完。
  val storeDoneMt       = RegInit(0.U(countWidth.W))
  val storeDoneNt       = RegInit(0.U(countWidth.W))
  val storeDoneRowIdx   = RegInit(0.U(5.W))

  // 发放游标在描述符进入 Ctrl 的响应队列时前进。它可领先完成游标，使 Store
  // 能连续取得后续行的地址信息而不等待任何 bank 写响应。
  val storeMetaMt       = RegInit(0.U(countWidth.W))
  val storeMetaNt       = RegInit(0.U(countWidth.W))
  val storeMetaRowIdx   = RegInit(0.U(5.W))
  val storeMetaBlockCount = RegInit(0.U(countWidth.W))

  // 描述符直接进入 Store 持有的两项地址 FIFO；这里仅记录已被 Store 接收、
  // 但尚未完成真实 bank 写回的行数。
  val storeOutstanding = RegInit(0.U(5.W))

  // 指令编码：rs2 携带 M/N/K/mode，rs1 携带 A/B/C 的 bank 与起始地址。
  val rs1 = issueQ.io.deq.bits.cmd.rs1
  val rs2 = issueQ.io.deq.bits.cmd.rs2

  val issueM    = rs2(11, 0)
  val issueN    = rs2(23, 12)
  val issueK    = rs2(35, 24)
  val issueMode = rs2(36)

  // issueQ 队首命令的组合解码；仅在开始或替换当前发射任务时锁存到 task。
  val nextTask = WireDefault(0.U.asTypeOf(new SystolicTask(b)))
  nextTask.mode       := issueMode
  nextTask.m          := issueM
  nextTask.n          := issueN
  nextTask.k          := issueK
  nextTask.op1_bank   := fitTo(rs1(9, 0), bankWidth)
  nextTask.op2_bank   := fitTo(rs1(19, 10), bankWidth)
  nextTask.wr_bank    := fitTo(rs1(29, 20), bankWidth)
  nextTask.op1_base   := fitTo(rs1(36, 30), addrWidth)
  nextTask.op2_base   := fitTo(rs1(43, 37), addrWidth)
  nextTask.wr_base    := fitTo(rs1(50, 44), addrWidth)
  nextTask.rob_id     := issueQ.io.deq.bits.rob_id
  nextTask.is_sub     := issueQ.io.deq.bits.is_sub
  nextTask.sub_rob_id := issueQ.io.deq.bits.sub_rob_id

  val currentKind = Wire(UInt(2.W))
  // EX 根据该标记选择新建累加上下文或继续已有的 K 维累加链。
  val currentKTileKind = Mux(
    kTiles === 1.U,
    SystolicKTileKind.DIRECT,
    Mux(kt === 0.U, SystolicKTileKind.FIRST,
      Mux(kt === (kTiles - 1.U), SystolicKTileKind.LAST, SystolicKTileKind.MIDDLE))
  )
  val currentValidM = validTileExtent(task.m, mt)
  val currentValidN = validTileExtent(task.n, nt)
  val currentValidK = validTileExtent(task.k, kt)
  // A 按 (M tile, K tile) 排列，B 按 (N tile, K tile) 排列；无论边缘 tile
  // 有多少有效行，每个 tile 的起始地址都固定间隔 16 行。
  val aRowOffset =
    (mt.pad(countWidth) * kTiles.pad(countWidth) + kt.pad(countWidth)) << log2Ceil(tile)
  val bRowOffset =
    (nt.pad(countWidth) * kTiles.pad(countWidth) + kt.pad(countWidth)) << log2Ceil(tile)
  // 当前 WS batch 的开区间上界，尾批可能不足 8 个 M tile。
  val wsBatchLimit = Mux(
    wsBatchBase + SystolicArrayConst.WsReuseTiles.U < mTiles,
    wsBatchBase + SystolicArrayConst.WsReuseTiles.U,
    mTiles)
  val wsReuseCount = wsBatchLimit - wsBatchBase
  val wsHasNextWeight = kt + 1.U < kTiles ||
    wsBatchBase + SystolicArrayConst.WsReuseTiles.U < mTiles || nt + 1.U < nTiles
  val wsPrefetchB = task.mode && !wsLoadPeB && wsReuseCount >= 3.U &&
    mt + 2.U === wsBatchLimit && wsHasNextWeight
  val prefetchNt = Wire(UInt(countWidth.W))
  val prefetchKt = Wire(UInt(countWidth.W))
  prefetchNt := nt
  prefetchKt := 0.U
  when(kt + 1.U < kTiles) {
    prefetchKt := kt + 1.U
  }.elsewhen(wsBatchBase + SystolicArrayConst.WsReuseTiles.U < mTiles) {
    prefetchKt := 0.U
  }.otherwise {
    prefetchNt := nt + 1.U
  }
  val prefetchValidN = validTileExtent(task.n, prefetchNt)
  val prefetchValidK = validTileExtent(task.k, prefetchKt)
  val prefetchBRowOffset =
    (prefetchNt.pad(countWidth) * kTiles.pad(countWidth) + prefetchKt.pad(countWidth)) << log2Ceil(tile)
  val selectedBRowOffset = Mux(wsPrefetchB, prefetchBRowOffset, bRowOffset)
  currentKind := Mux(
    task.mode,
    Mux(wsLoadPeB, SystolicCtrlLoadReqKind.READ_A_B_PE,
      Mux(wsPrefetchB, SystolicCtrlLoadReqKind.READ_A_B_BUF, SystolicCtrlLoadReqKind.READ_A_ONLY)),
    SystolicCtrlLoadReqKind.READ_AB
  )
  // retireQ 队首是当前唯一可完成的任务。storeMetaQ 独立保存地址发放顺序，
  // 允许后一任务的描述符在前一任务写回完成前进入 Store，同时保持全局保序。
  val storeTask = retireQ.io.deq.bits
  val storeMTiles = ceilDiv16(storeTask.m)
  val storeNTiles = ceilDiv16(storeTask.n)
  val storeMetaTask = storeMetaQ.io.deq.bits
  val storeMetaMTiles = ceilDiv16(storeMetaTask.m)
  val storeMetaNTiles = ceilDiv16(storeMetaTask.n)

  // 发放游标决定描述符中的地址。OS 使用连续的 16 元素 C block 计数；WS 按
  // 最终 C 布局重映射当前 (M tile, N tile, tile 内行)。这两类地址均不能等待
  // store_done_i 才前进，否则预取的下一行会重复前一行地址。
  val storeMetaRemainingCols = storeMetaTask.n.pad(countWidth) - (storeMetaNt << log2Ceil(tile))
  val storeMetaValidCols = Mux(
    storeMetaRemainingCols >= tile.U,
    tile.U(5.W),
    fitTo(storeMetaRemainingCols, 5))
  val storeMetaValidRows = validTileExtent(storeMetaTask.m, storeMetaMt)
  val storeMetaWsCompactBlockBase =
    (storeMetaMt.pad(countWidth) << log2Ceil(tile)) * storeMetaNTiles.pad(countWidth) +
      storeMetaNt.pad(countWidth) * storeMetaValidRows.pad(countWidth) +
      storeMetaRowIdx.pad(countWidth)
  val storeMetaBlockBase = Mux(
    storeMetaTask.mode,
    storeMetaWsCompactBlockBase,
    storeMetaBlockCount)

  // 完成游标只用于消费实际完成的行以及生成命令完成响应。它保持落后于发放游标，
  // 因而不会影响已经预取的地址。
  val storeDoneValidRows = validTileExtent(storeTask.m, storeDoneMt)
  val storeDoneLastTile = storeDoneMt === (storeMTiles - 1.U) &&
    storeDoneNt === (storeNTiles - 1.U)
  val storeDoneLastRow = storeDoneRowIdx + 1.U >= storeDoneValidRows
  val storeDoneTaskDone = storeDoneLastTile && storeDoneLastRow

  // 一次 Store 行请求对应的地址元数据。发放游标只遍历合法 M 行，因此每个
  // 描述符都对应一次真实 bank 写；N 维尾部由 row_valid_elems 的 byte mask 处理。
  val nextStoreResp = WireDefault(0.U.asTypeOf(new SystolicStoreCtrlResp(b)))
  nextStoreResp.row_valid_elems := storeMetaValidCols
  nextStoreResp.rob_id           := storeMetaTask.rob_id
  nextStoreResp.wr_bank          := storeMetaTask.wr_bank
  nextStoreResp.wr_group_base    := cGroupBase(storeMetaTask.wr_base, storeMetaBlockBase)
  nextStoreResp.wr_row_addr      := cRowAddr(storeMetaTask.wr_base, storeMetaBlockBase)

  val storeMetaLastTile = storeMetaMt === (storeMetaMTiles - 1.U) &&
    storeMetaNt === (storeMetaNTiles - 1.U)
  val storeMetaLastRow = storeMetaRowIdx + 1.U >= storeMetaValidRows
  val storeMetaTaskDone = storeMetaLastTile && storeMetaLastRow

  // 当前发射任务的最后一个 tile 被 Load 接收时，若 issueQ 已有后继任务，
  // 同一时钟沿装入后继任务，下一周期立即发送它的第一个 tile。
  val issueLastTile = issueActive && mt === (mTiles - 1.U) &&
    nt === (nTiles - 1.U) && kt === (kTiles - 1.U)
  val queuedTaskReady = issueQ.io.deq.valid && retireQ.io.enq.ready && storeMetaQ.io.enq.ready
  val canStartIssueTask = !issueActive && queuedTaskReady
  val replaceIssueTask = io.ctrl_ld_o.fire && issueLastTile && queuedTaskReady
  val acceptIssueTask = canStartIssueTask || replaceIssueTask

  issueQ.io.deq.ready := acceptIssueTask
  retireQ.io.enq.valid := acceptIssueTask
  retireQ.io.enq.bits.mode       := nextTask.mode
  retireQ.io.enq.bits.m          := nextTask.m
  retireQ.io.enq.bits.n          := nextTask.n
  retireQ.io.enq.bits.rob_id     := nextTask.rob_id
  retireQ.io.enq.bits.is_sub     := nextTask.is_sub
  retireQ.io.enq.bits.sub_rob_id := nextTask.sub_rob_id
  storeMetaQ.io.enq.valid := acceptIssueTask
  storeMetaQ.io.enq.bits.mode    := nextTask.mode
  storeMetaQ.io.enq.bits.m       := nextTask.m
  storeMetaQ.io.enq.bits.n       := nextTask.n
  storeMetaQ.io.enq.bits.wr_bank := nextTask.wr_bank
  storeMetaQ.io.enq.bits.wr_base := nextTask.wr_base
  storeMetaQ.io.enq.bits.rob_id  := nextTask.rob_id

  // 发射阶段只依赖 Load 的反压，不依赖 retireQ 队首任务的 Store 进度。
  io.ctrl_ld_o.valid := issueActive
  io.ctrl_ld_o.bits.req_kind     := currentKind
  io.ctrl_ld_o.bits.k_tile_kind  := currentKTileKind
  io.ctrl_ld_o.bits.acc_slot     := Mux(
    task.mode,
    fitTo(mt - wsBatchBase, log2Ceil(SystolicArrayConst.WsReuseTiles)),
    0.U)
  io.ctrl_ld_o.bits.valid_m      := currentValidM
  io.ctrl_ld_o.bits.valid_n      := currentValidN
  io.ctrl_ld_o.bits.valid_k      := currentValidK
  io.ctrl_ld_o.bits.b_valid_n    := Mux(wsPrefetchB, prefetchValidN, currentValidN)
  io.ctrl_ld_o.bits.b_valid_k    := Mux(wsPrefetchB, prefetchValidK, currentValidK)
  io.ctrl_ld_o.bits.weight_generation := wsWeightGeneration
  io.ctrl_ld_o.bits.op1_bank     := task.op1_bank
  io.ctrl_ld_o.bits.op1_group    := abGroup(task.op1_base, aRowOffset)
  io.ctrl_ld_o.bits.op1_row_base := abRow(task.op1_base, aRowOffset)
  io.ctrl_ld_o.bits.op2_bank     := task.op2_bank
  io.ctrl_ld_o.bits.op2_group    := abGroup(task.op2_base, selectedBRowOffset)
  io.ctrl_ld_o.bits.op2_row_base := abRow(task.op2_base, selectedBRowOffset)

  // 任务只在其最后一行已经实际写完后进入完成队列；前端对 cmdResp 的背压
  // 不会阻塞 issueTask 的 tile 发射，只会在完成队列满时停住 Store 的最后一行。
  io.cmdResp_o.valid := completeQ.io.deq.valid
  io.cmdResp_o.bits  := completeQ.io.deq.bits
  completeQ.io.deq.ready := io.cmdResp_o.ready
  completeQ.io.enq.valid := false.B
  completeQ.io.enq.bits.rob_id := storeTask.rob_id
  completeQ.io.enq.bits.is_sub := storeTask.is_sub
  completeQ.io.enq.bits.sub_rob_id := storeTask.sub_rob_id
  retireQ.io.deq.ready := false.B

  io.busy_o := issueActive || issueQ.io.deq.valid || retireQ.io.deq.valid ||
    completeQ.io.deq.valid || storeOutstanding =/= 0.U
  // Ctrl 不等待 Store 完成：只要元数据任务队列仍有下一行描述符，就保持 valid。
  // 最后一行 fire 后立即切换到下一任务，Store 依靠 FIFO 顺序和逐行 rob_id 保序。
  val completionSlotAvailable = completionReservations < 4.U || completeQ.io.deq.fire
  val storeCanEnqueue = storeMetaQ.io.deq.valid &&
    (!storeMetaTaskDone || completionSlotAvailable)
  io.store_ctrl_resp_o.valid := storeCanEnqueue
  io.store_ctrl_resp_o.bits  := nextStoreResp

  val storeMetaIssue = io.store_ctrl_resp_o.fire
  val reserveCompletion = storeMetaIssue && storeMetaTaskDone
  storeMetaQ.io.deq.ready := storeMetaIssue && storeMetaTaskDone
  switch(Cat(reserveCompletion, completeQ.io.deq.fire)) {
    is("b10".U) { completionReservations := completionReservations + 1.U }
    is("b01".U) { completionReservations := completionReservations - 1.U }
  }
  when(storeMetaIssue) {
    // 描述符已被 Store 接收，推进独立的地址发放游标。完成游标仍只响应
    // store_done_i，因此地址发放可以领先多个严格保序的任务。
    when(storeMetaTaskDone) {
      storeMetaMt := 0.U
      storeMetaNt := 0.U
      storeMetaRowIdx := 0.U
      storeMetaBlockCount := 0.U
    }.otherwise {
      storeMetaBlockCount := storeMetaBlockCount + 1.U

      storeMetaRowIdx := storeMetaRowIdx + 1.U
      when(storeMetaLastRow) {
        storeMetaRowIdx := 0.U

        when(storeMetaTask.mode) {
          when(storeMetaMt + 1.U < storeMetaMTiles) {
            storeMetaMt := storeMetaMt + 1.U
          }.otherwise {
            storeMetaMt := 0.U
            when(storeMetaNt + 1.U < storeMetaNTiles) {
              storeMetaNt := storeMetaNt + 1.U
            }
          }
        }.otherwise {
          when(storeMetaNt + 1.U < storeMetaNTiles) {
            storeMetaNt := storeMetaNt + 1.U
          }.otherwise {
            storeMetaNt := 0.U
            when(storeMetaMt + 1.U < storeMetaMTiles) {
              storeMetaMt := storeMetaMt + 1.U
            }
          }
        }
      }
    }
  }
  assert(completionReservations <= 4.U,
    "SystolicArrayCtrl: completion reservation overflow")

  // 每个被 Store 接收的描述符，最终都必须对应一个按顺序到达的 store_done_i。
  // Store 的地址 FIFO 和 Unit 的写飞行队列允许多行已发行但仍在 bank 中飞行。
  switch(Cat(io.store_ctrl_resp_o.fire, io.store_done_i)) {
    is("b10".U) { storeOutstanding := storeOutstanding + 1.U }
    is("b01".U) { storeOutstanding := storeOutstanding - 1.U }
  }

  when(io.store_done_i) {
    assert(storeOutstanding =/= 0.U, "SystolicArrayCtrl: store_done without an issued descriptor")

    when(storeDoneTaskDone) {
      assert(completeQ.io.enq.ready,
        "SystolicArrayCtrl: final store row completed without completion queue space")
      completeQ.io.enq.valid := true.B
      retireQ.io.deq.ready := completeQ.io.enq.ready
      storeDoneMt := 0.U
      storeDoneNt := 0.U
      storeDoneRowIdx := 0.U
    }.otherwise {
      storeDoneRowIdx := storeDoneRowIdx + 1.U

      // 一个输出 tile 的有效行完成后，按模式选择下一个 tile 的遍历顺序：
      // OS 为 N 内层；WS 为 M 内层。两者最终都映射到相同的 C 存储布局。
      when(storeDoneLastRow) {
        storeDoneRowIdx := 0.U

        when(storeTask.mode) {
          when(storeDoneMt + 1.U < storeMTiles) {
            storeDoneMt := storeDoneMt + 1.U
          }.otherwise {
            storeDoneMt := 0.U
            when(storeDoneNt + 1.U < storeNTiles) {
              storeDoneNt := storeDoneNt + 1.U
            }
          }
        }.otherwise {
          when(storeDoneNt + 1.U < storeNTiles) {
            storeDoneNt := storeDoneNt + 1.U
          }.otherwise {
            storeDoneNt := 0.U
            when(storeDoneMt + 1.U < storeMTiles) {
              storeDoneMt := storeDoneMt + 1.U
            }
          }
        }
      }
    }
  }
  assert(storeOutstanding <= 8.U, "SystolicArrayCtrl: Store completion pipeline overflow")

  // 防御性检查：每次实际发射时 K-tile 分类必须与 kt/kTiles 一致。
  when(io.ctrl_ld_o.fire) {
    when(kTiles === 1.U) {
      assert(currentKTileKind === SystolicKTileKind.DIRECT)
    }.elsewhen(kt === 0.U) {
      assert(currentKTileKind === SystolicKTileKind.FIRST)
    }.elsewhen(kt === (kTiles - 1.U)) {
      assert(currentKTileKind === SystolicKTileKind.LAST)
    }.otherwise {
      assert(currentKTileKind === SystolicKTileKind.MIDDLE)
    }

    when(issueLastTile) {
      when(!replaceIssueTask) {
        issueActive := false.B
      }
    }.otherwise {
      when(task.mode) {
        advanceIssueWsLoad()
      }.otherwise {
        advanceIssueOsTile()
      }
    }
  }

  /** OS 发射顺序为 K 内层、N 次之、M 外层：每个输出 tile 完成整条 K 累加链。 */
  private def advanceIssueOsTile(): Unit = {
    when(kt + 1.U < kTiles) {
      kt := kt + 1.U
    }.otherwise {
      kt := 0.U
      when(nt + 1.U < nTiles) {
        nt := nt + 1.U
      }.otherwise {
        nt := 0.U
        when(mt + 1.U < mTiles) {
          mt := mt + 1.U
        }.otherwise {
          assert(false.B, "SystolicArrayCtrl: advanced past final OS tile")
        }
      }
    }
  }

  /**
    * WS 发射顺序。
    *
    * 对固定 (N,K) 与一个最多 8 个 M tile 的 batch，先为 batch 首个 M tile
    * 发送 READ_A_B_PE 装载权重，随后为其余 M tile 发送 READ_A_ONLY。所有 K
    * tile 完成后才切换下一 batch 或下一 N tile。
    */
  private def advanceIssueWsLoad(): Unit = {
    when(mt + 1.U < wsBatchLimit) {
      mt := mt + 1.U
      wsLoadPeB := false.B
      when(wsPrefetchB) {
        wsPrefetchedWeights := true.B
      }
    }.otherwise {
      mt := wsBatchBase
      when(kt + 1.U < kTiles) {
        kt := kt + 1.U
        wsLoadPeB := !wsPrefetchedWeights
        wsWeightGeneration := !wsWeightGeneration
        wsPrefetchedWeights := false.B
      }.otherwise {
        kt := 0.U
        when(wsBatchBase + SystolicArrayConst.WsReuseTiles.U < mTiles) {
          wsBatchBase := wsBatchBase + SystolicArrayConst.WsReuseTiles.U
          mt := wsBatchBase + SystolicArrayConst.WsReuseTiles.U
          wsLoadPeB := !wsPrefetchedWeights
          wsWeightGeneration := !wsWeightGeneration
          wsPrefetchedWeights := false.B
        }.otherwise {
          wsBatchBase := 0.U
          mt := 0.U
          when(nt + 1.U < nTiles) {
            nt := nt + 1.U
            wsLoadPeB := !wsPrefetchedWeights
            wsWeightGeneration := !wsWeightGeneration
            wsPrefetchedWeights := false.B
          }.otherwise {
            assert(false.B, "SystolicArrayCtrl: advanced past final WS tile")
          }
        }
      }
    }
  }

  /** 接受队首命令并仅复位 Load 发射游标；Store 地址与完成游标独立推进。 */
  private def startIssueTask(): Unit = {
    task   := nextTask
    mTiles := ceilDiv16(issueM)
    nTiles := ceilDiv16(issueN)
    kTiles := ceilDiv16(issueK)
    mt := 0.U
    nt := 0.U
    kt := 0.U
    wsBatchBase := 0.U
    wsLoadPeB := issueMode
    wsWeightGeneration := false.B
    wsPrefetchedWeights := false.B
    issueActive := true.B

    assert(issueM =/= 0.U && issueN =/= 0.U && issueK =/= 0.U, "SystolicArrayCtrl: M/N/K must be non-zero")
  }

  when(acceptIssueTask) {
    startIssueTask()
  }
}
