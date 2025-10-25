package framework.builtin.frontend.globalrs

import chisel3._
import chisel3.util._
import chisel3.experimental._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import framework.builtin.frontend.PostGDCmd

class GlobalROB(implicit b: CustomBuckyBallConfig, p: Parameters) extends Module {
  val io = IO(new Bundle {
    // 分配接口
    val alloc = Flipped(new DecoupledIO(new PostGDCmd))

    // 发射接口 - 发射未完成的头部指令
    val issue = new DecoupledIO(new GlobalRobEntry)

    // 完成接口 - 报告指令完成
    val complete = Flipped(new DecoupledIO(UInt(log2Up(b.rob_entries).W)))

    // 状态信号 - 暴露给保留站用于决策
    val empty = Output(Bool())
    val full  = Output(Bool())
    val head_ptr = Output(UInt(log2Up(b.rob_entries).W))         // head指针位置
    val issued_count = Output(UInt(log2Up(b.rob_entries + 1).W)) // 已发射未完成的指令数
    val entry_valid = Output(Vec(b.rob_entries, Bool()))          // 每个条目是否有效
    val entry_complete = Output(Vec(b.rob_entries, Bool()))       // 每个条目是否完成
  })

  // 环状ROB结构
  val robEntries = Reg(Vec(b.rob_entries, new GlobalRobEntry))
  val robValid   = RegInit(VecInit(Seq.fill(b.rob_entries)(false.B)))  // 条目是否有效
  val robIssued  = RegInit(VecInit(Seq.fill(b.rob_entries)(false.B)))  // 条目是否已发射
  val robComplete = RegInit(VecInit(Seq.fill(b.rob_entries)(false.B))) // 条目是否已完成

  // 循环队列指针
  val headPtr = RegInit(0.U(log2Up(b.rob_entries).W))  // 指向最老的未提交指令
  val tailPtr = RegInit(0.U(log2Up(b.rob_entries).W))  // 指向下一个要分配的位置
  val robIdCounter = RegInit(0.U(log2Up(b.rob_entries).W))  // ROB ID循环计数器

  // 已发射但未完成的指令数量（用于限制发射）
  val issuedCount = RegInit(0.U(log2Up(b.rob_entries + 1).W))
  val maxIssueLimit = (b.rob_entries / 2).U  // 最多发射ROB深度一半的指令

  // 队列状态
  val isEmpty = headPtr === tailPtr && !robValid(headPtr)
  val isFull  = headPtr === tailPtr && robValid(headPtr)

// -----------------------------------------------------------------------------
// 入站 - 指令分配
// -----------------------------------------------------------------------------
  io.alloc.ready := !isFull

  when(io.alloc.fire) {
    robEntries(tailPtr).cmd    := io.alloc.bits
    robEntries(tailPtr).rob_id := robIdCounter
    robValid(tailPtr)   := true.B
    robIssued(tailPtr)  := false.B
    robComplete(tailPtr) := false.B

    // 更新tail指针和rob_id计数器（循环）
    tailPtr := Mux(tailPtr === (b.rob_entries - 1).U, 0.U, tailPtr + 1.U)
    robIdCounter := Mux(robIdCounter === (b.rob_entries - 1).U, 0.U, robIdCounter + 1.U)
  }

// -----------------------------------------------------------------------------
// 完成信号处理
// -----------------------------------------------------------------------------
  io.complete.ready := true.B
  when(io.complete.fire) {
    val completeId = io.complete.bits
    robComplete(completeId) := true.B
    // 完成时，已发射计数减1
    when(robIssued(completeId)) {
      issuedCount := issuedCount - 1.U
    }
  }

// -----------------------------------------------------------------------------
// 出站 - 顺序发射指令（从head开始）
// -----------------------------------------------------------------------------
  // 查找从head开始第一个有效且未发射的指令
  val canIssue = Wire(Bool())
  val issuePtr = Wire(UInt(log2Up(b.rob_entries).W))

  // 默认值
  canIssue := false.B
  issuePtr := headPtr

  // 从head开始扫描，找到第一个可发射的指令
  val scanValid = Wire(Vec(b.rob_entries, Bool()))
  for (i <- 0 until b.rob_entries) {
    val ptr = Mux(headPtr + i.U >= b.rob_entries.U,
                  headPtr + i.U - b.rob_entries.U,
                  headPtr + i.U)
    scanValid(i) := robValid(ptr) && !robIssued(ptr) && !robComplete(ptr)
  }

  // 找到第一个可发射的位置
  val firstValid = PriorityEncoder(scanValid.asUInt)
  val hasValid = scanValid.asUInt.orR

  val actualIssuePtr = Mux(headPtr + firstValid >= b.rob_entries.U,
                           headPtr + firstValid - b.rob_entries.U,
                           headPtr + firstValid)

  // 只有在未达到发射限制时才能发射
  val canIssueMore = issuedCount < maxIssueLimit
  canIssue := hasValid && canIssueMore
  issuePtr := actualIssuePtr

  io.issue.valid := canIssue
  io.issue.bits  := robEntries(issuePtr)

  when(io.issue.fire) {
    robIssued(issuePtr) := true.B
    issuedCount := issuedCount + 1.U
  }

// -----------------------------------------------------------------------------
// 指令提交 - 乱序提交所有已完成的指令
// -----------------------------------------------------------------------------
  // 提交所有已完成的指令
  for (i <- 0 until b.rob_entries) {
    when(robValid(i.U) && robComplete(i.U)) {
      robValid(i.U) := false.B
      robIssued(i.U) := false.B
      robComplete(i.U) := false.B
    }
  }

  // 更新head指针：跳过所有已完成（即将被清除）的位置
  // 找到从head开始第一个"有效且未完成"的指令位置
  val nextHeadCandidates = Wire(Vec(b.rob_entries, Bool()))
  for (i <- 0 until b.rob_entries) {
    val ptr = Mux(headPtr + i.U >= b.rob_entries.U,
                  headPtr + i.U - b.rob_entries.U,
                  headPtr + i.U)
    // 条目有效且未完成（不会被提交）
    nextHeadCandidates(i) := robValid(ptr) && !robComplete(ptr)
  }

  val hasUncommitted = nextHeadCandidates.asUInt.orR
  val nextHeadOffset = PriorityEncoder(nextHeadCandidates.asUInt)
  val nextHeadPtr = Mux(headPtr + nextHeadOffset >= b.rob_entries.U,
                        headPtr + nextHeadOffset - b.rob_entries.U,
                        headPtr + nextHeadOffset)

  // 更新head指针：
  // - 如果有未完成的指令，移动head到第一个未完成的位置
  // - 如果没有未完成的指令（全部完成），head移动到tail（ROB为空）
  headPtr := Mux(hasUncommitted, nextHeadPtr, tailPtr)

// -----------------------------------------------------------------------------
// 状态信号 - 暴露给保留站
// -----------------------------------------------------------------------------
  io.empty := isEmpty
  io.full  := isFull
  io.head_ptr := headPtr
  io.issued_count := issuedCount
  io.entry_valid := robValid
  io.entry_complete := robComplete
}
