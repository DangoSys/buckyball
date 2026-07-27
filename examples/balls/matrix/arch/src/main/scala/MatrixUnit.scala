package examples.balls.matrix

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.balldomain.rs.{BallRsComplete, BallRsIssue}
import framework.balldomain.blink.{BallStatus, BankRead, BankWrite}
import framework.top.GlobalConfig
import framework.balldomain.prototype.systolicarray.{
  SystolicArrayConst,
  SystolicArrayCtrl,
  SystolicArrayEX,
  SystolicArrayLoad,
  SystolicArrayStore,
  SystolicStoreWriteReq
}

@instantiable
class MatrixUnit(val b: GlobalConfig) extends Module {
  private val accElemBits   = SystolicArrayConst.AccElemBits
  private val writePorts    = SystolicArrayConst.StoreWritePorts
  private val elemsPerPort  = SystolicArrayConst.StorePortElemCount
  private val resultRowBits = SystolicArrayConst.ResultRowBits
  private val groupWidth    = log2Up(b.memDomain.bankNum)
  private val writeDataEntries = 2
  private val writeDataIdxWidth = log2Ceil(writeDataEntries)
  private val writeDataCountWidth = log2Ceil(writeDataEntries + 1)
  private val writeTrackEntries = 4
  private val writeTrackIdxWidth = log2Ceil(writeTrackEntries)
  private val writeTrackCountWidth = log2Ceil(writeTrackEntries + 1)

  private def portMask(validElems: UInt): UInt =
    VecInit((0 until writePorts).map(port =>
      (port * elemsPerPort).U < validElems)).asUInt

  private val ballMapping = b.ballDomain.ballIdMappings.find(_.ballName == "MatrixBall")
    .getOrElse(throw new IllegalArgumentException("MatrixBall not found in config"))
  private val inBW  = ballMapping.inBW
  private val outBW = ballMapping.outBW

  require(inBW >= 2, "MatrixUnit requires at least two read ports for op1/op2")
  require(outBW >= writePorts, "MatrixUnit requires four write ports for one 16xi32 C row per Store write")
  require(b.memDomain.bankWidth == 128, "MatrixUnit expects 128-bit physical bank rows")
  require(b.memDomain.bankMaskLen == 16, "MatrixUnit expects byte write masks on 128-bit bank rows")
  require(resultRowBits == writePorts * b.memDomain.bankWidth)

  @public
  val io = IO(new Bundle {
    val cmdReq    = Flipped(Decoupled(new BallRsIssue(b)))
    val cmdResp   = Decoupled(new BallRsComplete(b))
    val bankRead  = Vec(inBW, Flipped(new BankRead(b)))
    val bankWrite = Vec(outBW, Flipped(new BankWrite(b)))
    val status    = new BallStatus
  })
  val resetActive = reset.asBool

  val ctrl:  Instance[SystolicArrayCtrl]  = Instantiate(new SystolicArrayCtrl(b))
  val load:  Instance[SystolicArrayLoad]  = Instantiate(new SystolicArrayLoad(b))
  val ex:    Instance[SystolicArrayEX]    = Instantiate(new SystolicArrayEX(b))
  val store: Instance[SystolicArrayStore] = Instantiate(new SystolicArrayStore(b))

  ctrl.io.cmdReq.valid := io.cmdReq.valid && !resetActive
  ctrl.io.cmdReq.bits  := io.cmdReq.bits
  io.cmdReq.ready      := ctrl.io.cmdReq.ready && !resetActive

  io.cmdResp.valid          := ctrl.io.cmdResp_o.valid && !resetActive
  io.cmdResp.bits           := ctrl.io.cmdResp_o.bits
  ctrl.io.cmdResp_o.ready   := io.cmdResp.ready && !resetActive

  load.io.ctrl_ld_i.valid := ctrl.io.ctrl_ld_o.valid
  load.io.ctrl_ld_i.bits  := ctrl.io.ctrl_ld_o.bits
  ctrl.io.ctrl_ld_o.ready := load.io.ctrl_ld_i.ready

  ex.io.load_ex_req_kind    := load.io.load_ex_req_kind
  ex.io.load_ex_k_tile_kind := load.io.load_ex_k_tile_kind
  ex.io.load_ex_acc_slot    := load.io.load_ex_acc_slot
  ex.io.load_ex_valid_m     := load.io.load_ex_valid_m
  ex.io.load_ex_valid_n     := load.io.load_ex_valid_n
  ex.io.load_ex_valid_k     := load.io.load_ex_valid_k
  ex.io.load_ex_b_valid_n   := load.io.load_ex_b_valid_n
  ex.io.load_ex_b_valid_k   := load.io.load_ex_b_valid_k
  ex.io.load_ex_weight_generation := load.io.load_ex_weight_generation
  ex.io.load_ex_op1_i.valid := load.io.load_ex_op1_o.valid
  ex.io.load_ex_op1_i.bits  := load.io.load_ex_op1_o.bits
  load.io.load_ex_op1_o.ready := ex.io.load_ex_op1_i.ready
  ex.io.load_ex_op2_i.valid := load.io.load_ex_op2_o.valid
  ex.io.load_ex_op2_i.bits  := load.io.load_ex_op2_o.bits
  load.io.load_ex_op2_o.ready := ex.io.load_ex_op2_i.ready

  store.io.ex_st_i.valid := ex.io.ex_st_o.valid
  store.io.ex_st_i.bits  := ex.io.ex_st_o.bits
  ex.io.ex_st_o.ready    := store.io.ex_st_i.ready
  store.io.store_ctrl_resp_i.valid := ctrl.io.store_ctrl_resp_o.valid
  store.io.store_ctrl_resp_i.bits  := ctrl.io.store_ctrl_resp_o.bits
  ctrl.io.store_ctrl_resp_o.ready  := store.io.store_ctrl_resp_i.ready
  ctrl.io.store_done_i := store.io.store_done_o

  for (i <- 0 until inBW) {
    io.bankRead(i).rob_id  := 0.U
    io.bankRead(i).ball_id := 0.U
    io.bankRead(i).bank_id := 0.U
    io.bankRead(i).group_id := 0.U
    io.bankRead(i).io.req.valid := false.B
    io.bankRead(i).io.req.bits.addr := 0.U
    io.bankRead(i).io.resp.ready := false.B
    load.io.bankReadReq(i).ready := false.B
    load.io.bankReadResp(i).valid := false.B
    load.io.bankReadResp(i).bits := 0.U.asTypeOf(load.io.bankReadResp(i).bits)
  }

  io.bankRead(0).bank_id := load.io.op1_rd_bank_o
  io.bankRead(0).group_id := load.io.op1_rd_group_o
  io.bankRead(0).io.req.valid := load.io.bankReadReq(0).valid && !resetActive
  io.bankRead(0).io.req.bits  := load.io.bankReadReq(0).bits
  load.io.bankReadReq(0).ready := io.bankRead(0).io.req.ready && !resetActive
  load.io.bankReadResp(0).valid := io.bankRead(0).io.resp.valid && !resetActive
  load.io.bankReadResp(0).bits  := io.bankRead(0).io.resp.bits
  io.bankRead(0).io.resp.ready := load.io.bankReadResp(0).ready && !resetActive

  io.bankRead(1).bank_id := load.io.op2_rd_bank_o
  io.bankRead(1).group_id := load.io.op2_rd_group_o
  io.bankRead(1).io.req.valid := load.io.bankReadReq(1).valid && !resetActive
  io.bankRead(1).io.req.bits  := load.io.bankReadReq(1).bits
  load.io.bankReadReq(1).ready := io.bankRead(1).io.req.ready && !resetActive
  load.io.bankReadResp(1).valid := io.bankRead(1).io.resp.valid && !resetActive
  load.io.bankReadResp(1).bits  := io.bankRead(1).io.resp.bits
  io.bankRead(1).io.resp.ready := load.io.bankReadResp(1).ready && !resetActive

  // 两项数据队列只保留尚未完全发出的 C 行；请求全部发出后立即释放 512-bit 数据。
  // 四项轻量 tracker 继续等待 bank response，从而保持原有的最大在途行数。
  val writeData = Reg(Vec(writeDataEntries, new SystolicStoreWriteReq(b)))
  val writeDataTrack = Reg(Vec(writeDataEntries, UInt(writeTrackIdxWidth.W)))
  val writeDataReadPtr = RegInit(0.U(writeDataIdxWidth.W))
  val writeDataWritePtr = RegInit(0.U(writeDataIdxWidth.W))
  val writeDataCount = RegInit(0.U(writeDataCountWidth.W))

  val writeTrackValid = RegInit(VecInit(Seq.fill(writeTrackEntries)(false.B)))
  val writeTrackRequiredMask = RegInit(VecInit(Seq.fill(writeTrackEntries)(0.U(writePorts.W))))
  val writeTrackIssuedMask = RegInit(VecInit(Seq.fill(writeTrackEntries)(0.U(writePorts.W))))
  val writeTrackAckMask = RegInit(VecInit(Seq.fill(writeTrackEntries)(0.U(writePorts.W))))
  val writeTrackReadPtr = RegInit(0.U(writeTrackIdxWidth.W))
  val writeTrackWritePtr = RegInit(0.U(writeTrackIdxWidth.W))
  val writeTrackCount = RegInit(0.U(writeTrackCountWidth.W))

  val issueEntry = writeData(writeDataReadPtr)
  val issueEntryValid = writeDataCount =/= 0.U
  val issueTrackIdx = writeDataTrack(writeDataReadPtr)
  val issuePortMask = writeTrackRequiredMask(issueTrackIdx)

  // bank response 不携带 Unit 自定义的行标签；对每个写口，从最早 entry 开始寻找
  // 该口尚未确认的请求，即可按该口的 FIFO 顺序正确归属 response。
  val responseTarget = Wire(Vec(writePorts, UInt(writeTrackIdxWidth.W)))
  val responseTargetValid = Wire(Vec(writePorts, Bool()))
  for (port <- 0 until writePorts) {
    responseTarget(port) := writeTrackReadPtr
    var targetFound = false.B
    for (offset <- 0 until writeTrackEntries) {
      val candidateIdx = (writeTrackReadPtr + offset.U)(writeTrackIdxWidth - 1, 0)
      val candidateWaitsForPort = writeTrackValid(candidateIdx) &&
        writeTrackIssuedMask(candidateIdx)(port) && !writeTrackAckMask(candidateIdx)(port)
      when(!targetFound && candidateWaitsForPort) {
        responseTarget(port) := candidateIdx
      }
      targetFound = targetFound || candidateWaitsForPort
    }
    responseTargetValid(port) := targetFound
  }

  for (i <- 0 until outBW) {
    io.bankWrite(i).rob_id  := 0.U
    io.bankWrite(i).ball_id := 0.U
    io.bankWrite(i).bank_id := 0.U
    io.bankWrite(i).group_id := 0.U
    io.bankWrite(i).io.req.valid := false.B
    io.bankWrite(i).io.req.bits.addr := 0.U
    io.bankWrite(i).io.req.bits.data := 0.U
    io.bankWrite(i).io.req.bits.mask := VecInit(Seq.fill(b.memDomain.bankMaskLen)(false.B))
    io.bankWrite(i).io.resp.ready := false.B
  }

  for (port <- 0 until writePorts) {
    val byteBase = port * elemsPerPort
    val groupWithPort = issueEntry.wr_group_base +& port.U(groupWidth.W)

    io.bankWrite(port).rob_id := issueEntry.rob_id
    io.bankWrite(port).bank_id := issueEntry.wr_bank
    io.bankWrite(port).group_id := groupWithPort(groupWidth - 1, 0)
    io.bankWrite(port).io.req.valid := issueEntryValid && issuePortMask(port) &&
      !writeTrackIssuedMask(issueTrackIdx)(port) && !resetActive
    io.bankWrite(port).io.req.bits.addr := issueEntry.wr_row_addr
    io.bankWrite(port).io.req.bits.data := issueEntry.data(
      (port + 1) * b.memDomain.bankWidth - 1,
      port * b.memDomain.bankWidth)

    val mask = Wire(Vec(b.memDomain.bankMaskLen, Bool()))
    for (byte <- 0 until b.memDomain.bankMaskLen) {
      val logicalElem = (byteBase + byte / (accElemBits / 8)).U
      mask(byte) := logicalElem < issueEntry.valid_elems
    }
    io.bankWrite(port).io.req.bits.mask := mask
    io.bankWrite(port).io.resp.ready := responseTargetValid(port) && !resetActive
  }

  val issueFireMask = VecInit((0 until writePorts).map(port =>
    io.bankWrite(port).io.req.fire)).asUInt
  val issueMaskAfterFire = writeTrackIssuedMask(issueTrackIdx) | issueFireMask
  val issueEntryFinished = issueEntryValid &&
    (writeTrackIssuedMask(issueTrackIdx) & issuePortMask) =/= issuePortMask &&
    (issueMaskAfterFire & issuePortMask) === issuePortMask

  val responseFire = Wire(Vec(writePorts, Bool()))
  for (port <- 0 until writePorts) {
    responseFire(port) := responseTargetValid(port) && io.bankWrite(port).io.resp.valid && !resetActive
  }
  val responseAckMask = Wire(Vec(writeTrackEntries, UInt(writePorts.W)))
  for (entry <- 0 until writeTrackEntries) {
    responseAckMask(entry) := VecInit((0 until writePorts).map { port =>
      responseFire(port) && responseTarget(port) === entry.U(writeTrackIdxWidth.W)
    }).asUInt
  }

  // 只允许队首在其所需 port 都返回 response 后退休，因而 Store/Ctrl 看到的
  // wr_done_i 仍是一行一次且严格保序的完成脉冲。
  val headPortMask = writeTrackRequiredMask(writeTrackReadPtr)
  val headAckAfterResponse = writeTrackAckMask(writeTrackReadPtr) |
    responseAckMask(writeTrackReadPtr)
  val headComplete = writeTrackValid(writeTrackReadPtr) &&
    (writeTrackIssuedMask(writeTrackReadPtr) & headPortMask) === headPortMask &&
    (headAckAfterResponse & headPortMask) === headPortMask
  val canEnqueueData = writeDataCount < writeDataEntries.U || issueEntryFinished
  val canEnqueueTrack = writeTrackCount < writeTrackEntries.U || headComplete

  store.io.wr_o.ready := canEnqueueData && canEnqueueTrack
  store.io.wr_done_i := headComplete
  val writeEnqueue = store.io.wr_o.fire

  for (entry <- 0 until writeTrackEntries) {
    when(writeTrackValid(entry) && responseAckMask(entry).orR) {
      writeTrackAckMask(entry) := writeTrackAckMask(entry) | responseAckMask(entry)
    }
  }

  when(issueFireMask.orR) {
    writeTrackIssuedMask(issueTrackIdx) := issueMaskAfterFire
  }
  when(issueEntryFinished) {
    writeDataReadPtr := writeDataReadPtr + 1.U
  }
  when(headComplete) {
    writeTrackValid(writeTrackReadPtr) := false.B
    writeTrackReadPtr := writeTrackReadPtr + 1.U
  }
  when(writeEnqueue) {
    writeData(writeDataWritePtr) := store.io.wr_o.bits
    writeDataTrack(writeDataWritePtr) := writeTrackWritePtr
    writeDataWritePtr := writeDataWritePtr + 1.U

    writeTrackRequiredMask(writeTrackWritePtr) := portMask(store.io.wr_o.bits.valid_elems)
    writeTrackIssuedMask(writeTrackWritePtr) := 0.U
    writeTrackAckMask(writeTrackWritePtr) := 0.U
    writeTrackValid(writeTrackWritePtr) := true.B
    writeTrackWritePtr := writeTrackWritePtr + 1.U
  }
  switch(Cat(writeEnqueue, issueEntryFinished)) {
    is("b10".U) { writeDataCount := writeDataCount + 1.U }
    is("b01".U) { writeDataCount := writeDataCount - 1.U }
  }
  switch(Cat(writeEnqueue, headComplete)) {
    is("b10".U) { writeTrackCount := writeTrackCount + 1.U }
    is("b01".U) { writeTrackCount := writeTrackCount - 1.U }
  }

  when(writeEnqueue) {
    assert(portMask(store.io.wr_o.bits.valid_elems).orR,
      "MatrixUnit: write row has no valid port")
  }
  when(issueEntryValid) {
    assert(writeTrackValid(issueTrackIdx),
      "MatrixUnit: write data references an invalid response tracker")
  }
  assert(writeDataCount <= writeDataEntries.U,
    "MatrixUnit: write data queue overflow")
  assert(writeTrackCount <= writeTrackEntries.U,
    "MatrixUnit: write response tracker overflow")

  val hasInput  = RegInit(false.B)
  val hasOutput = RegInit(false.B)

  when(io.cmdReq.fire) {
    hasInput := true.B
  }
  when(io.cmdResp.fire) {
    hasOutput := false.B
    hasInput  := false.B
  }
  when(io.cmdResp.valid && !hasOutput) {
    hasOutput := true.B
  }

  io.status.idle    := resetActive || (!hasInput && !hasOutput && !ctrl.io.busy_o)
  io.status.running := !resetActive && ctrl.io.busy_o
}
