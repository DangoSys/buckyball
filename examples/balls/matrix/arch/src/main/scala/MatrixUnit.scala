package examples.balls.matrix

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.balldomain.rs.{BallRsComplete, BallRsIssue}
import framework.balldomain.blink.{BallStatus, BankRead, BankWrite}
import framework.top.GlobalConfig

@instantiable
class MatrixUnit(val b: GlobalConfig) extends Module {
  private val accElemBits   = MatrixConst.AccElemBits
  private val writePorts    = MatrixConst.StoreWritePorts
  private val elemsPerPort  = MatrixConst.StorePortElemCount
  private val resultRowBits = MatrixConst.ResultRowBits
  private val groupWidth    = log2Up(b.memDomain.bankNum)
  private val addrWidth     = log2Up(b.memDomain.bankEntries)

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

  val ctrl:  Instance[MatrixCtrl]  = Instantiate(new MatrixCtrl(b))
  val load:  Instance[MatrixLoad]  = Instantiate(new MatrixLoad(b))
  val ex:    Instance[MatrixEX]    = Instantiate(new MatrixEX(b))
  val store: Instance[MatrixStore] = Instantiate(new MatrixStore(b))

  ctrl.io.cmdReq.valid := io.cmdReq.valid
  ctrl.io.cmdReq.bits  := io.cmdReq.bits
  io.cmdReq.ready      := ctrl.io.cmdReq.ready

  io.cmdResp.valid          := ctrl.io.cmdResp_o.valid
  io.cmdResp.bits           := ctrl.io.cmdResp_o.bits
  ctrl.io.cmdResp_o.ready   := io.cmdResp.ready

  load.io.ctrl_ld_i.valid := ctrl.io.ctrl_ld_o.valid
  load.io.ctrl_ld_i.bits  := ctrl.io.ctrl_ld_o.bits
  ctrl.io.ctrl_ld_o.ready := load.io.ctrl_ld_i.ready

  ex.io.load_ex_req_kind    := load.io.load_ex_req_kind
  ex.io.load_ex_k_tile_kind := load.io.load_ex_k_tile_kind
  ex.io.load_ex_acc_slot    := load.io.load_ex_acc_slot
  ex.io.load_ex_valid_m     := load.io.load_ex_valid_m
  ex.io.load_ex_valid_n     := load.io.load_ex_valid_n
  ex.io.load_ex_valid_k     := load.io.load_ex_valid_k
  ex.io.load_ex_row_count   := load.io.load_ex_row_count
  ex.io.load_ex_first_row   := load.io.load_ex_first_row
  ex.io.load_ex_last_row    := load.io.load_ex_last_row
  ex.io.load_ex_op1_i.valid := load.io.load_ex_op1_o.valid
  ex.io.load_ex_op1_i.bits  := load.io.load_ex_op1_o.bits
  load.io.load_ex_op1_o.ready := ex.io.load_ex_op1_i.ready
  ex.io.load_ex_op2_i.valid := load.io.load_ex_op2_o.valid
  ex.io.load_ex_op2_i.bits  := load.io.load_ex_op2_o.bits
  load.io.load_ex_op2_o.ready := ex.io.load_ex_op2_i.ready

  store.io.ex_st_i.valid := ex.io.ex_st_o.valid
  store.io.ex_st_i.bits  := ex.io.ex_st_o.bits
  ex.io.ex_st_o.ready    := store.io.ex_st_i.ready
  ctrl.io.store_req_i := store.io.store_req_o
  store.io.store_ctrl_resp_i.valid := ctrl.io.store_ctrl_resp_o.valid
  store.io.store_ctrl_resp_i.bits  := ctrl.io.store_ctrl_resp_o.bits
  ctrl.io.store_ctrl_resp_o.ready  := store.io.store_ctrl_resp_i.ready
  ctrl.io.store_done_i := store.io.store_done_o

  for (i <- 0 until inBW) {
    io.bankRead(i).rob_id  := ctrl.io.active_rob_id_o
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
  io.bankRead(0).io.req.valid := load.io.bankReadReq(0).valid
  io.bankRead(0).io.req.bits  := load.io.bankReadReq(0).bits
  load.io.bankReadReq(0).ready := io.bankRead(0).io.req.ready
  load.io.bankReadResp(0).valid := io.bankRead(0).io.resp.valid
  load.io.bankReadResp(0).bits  := io.bankRead(0).io.resp.bits
  io.bankRead(0).io.resp.ready := load.io.bankReadResp(0).ready

  io.bankRead(1).bank_id := load.io.op2_rd_bank_o
  io.bankRead(1).group_id := load.io.op2_rd_group_o
  io.bankRead(1).io.req.valid := load.io.bankReadReq(1).valid
  io.bankRead(1).io.req.bits  := load.io.bankReadReq(1).bits
  load.io.bankReadReq(1).ready := io.bankRead(1).io.req.ready
  load.io.bankReadResp(1).valid := io.bankRead(1).io.resp.valid
  load.io.bankReadResp(1).bits  := io.bankRead(1).io.resp.bits
  io.bankRead(1).io.resp.ready := load.io.bankReadResp(1).ready

  for (i <- 0 until outBW) {
    io.bankWrite(i).rob_id  := ctrl.io.active_rob_id_o
    io.bankWrite(i).ball_id := 0.U
    io.bankWrite(i).bank_id := 0.U
    io.bankWrite(i).group_id := 0.U
    io.bankWrite(i).io.req.valid := false.B
    io.bankWrite(i).io.req.bits.addr := 0.U
    io.bankWrite(i).io.req.bits.data := 0.U
    io.bankWrite(i).io.req.bits.mask := VecInit(Seq.fill(b.memDomain.bankMaskLen)(false.B))
    io.bankWrite(i).io.resp.ready := false.B
  }

  val rowWriteActive = RegInit(false.B)
  val rowWriteReg    = RegInit(0.U.asTypeOf(new MatrixStoreWriteReq(b)))
  val writeBeatIdx   = RegInit(0.U(3.W))
  val writeWaiting   = RegInit(false.B)

  store.io.wr_o.ready := !rowWriteActive
  store.io.wr_done_i := false.B

  when(store.io.wr_o.fire) {
    rowWriteActive := true.B
    rowWriteReg    := store.io.wr_o.bits
    writeBeatIdx   := 0.U
    writeWaiting   := false.B
  }

  for (port <- 0 until writePorts) {
    val startElem = Mux(port.U === 0.U, writeBeatIdx * elemsPerPort.U,
                         0.U(5.W))
    val portData = (rowWriteReg.data >> (startElem * accElemBits.U))(
      b.memDomain.bankWidth - 1, 0)
    val targetGroup = rowWriteReg.wr_group_base + writeBeatIdx
    val targetAddr  = rowWriteReg.wr_row_addr / rowWriteReg.beat_count

    io.bankWrite(port).bank_id := rowWriteReg.wr_bank
    io.bankWrite(port).group_id := targetGroup(groupWidth - 1, 0)
    val requestValid = if (port == 0) {
      rowWriteActive && !writeWaiting && writeBeatIdx < rowWriteReg.beat_count
    } else {
      false.B
    }
    io.bankWrite(port).io.req.valid := requestValid
    io.bankWrite(port).io.req.bits.addr := targetAddr(addrWidth - 1, 0)
    io.bankWrite(port).io.req.bits.data := portData

    val mask = Wire(Vec(b.memDomain.bankMaskLen, Bool()))
    for (byte <- 0 until b.memDomain.bankMaskLen) {
      val logicalElem = writeBeatIdx * elemsPerPort.U +
        (byte / (accElemBits / 8)).U
      mask(byte) := logicalElem < rowWriteReg.valid_elems
    }
    io.bankWrite(port).io.req.bits.mask := mask

    val responseReady = if (port == 0) {
      rowWriteActive && writeWaiting
    } else {
      false.B
    }
    io.bankWrite(port).io.resp.ready := responseReady

    when(io.bankWrite(port).io.req.fire) {
      writeWaiting := true.B
    }
    when(io.bankWrite(port).io.resp.fire) {
      writeWaiting := false.B
      when(writeBeatIdx + 1.U >= rowWriteReg.beat_count) {
        rowWriteActive := false.B
        store.io.wr_done_i := true.B
      }.otherwise {
        writeBeatIdx := writeBeatIdx + 1.U
      }
    }
  }

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

  io.status.idle    := !hasInput && !hasOutput && !ctrl.io.busy_o
  io.status.running := ctrl.io.busy_o
}
