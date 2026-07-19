package examples.balls.matrix

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.top.GlobalConfig

@instantiable
class MatrixStore(val b: GlobalConfig) extends Module {
  private val tile          = MatrixConst.Tile
  private val accElemBits   = MatrixConst.AccElemBits
  private val resultRowBits = MatrixConst.ResultRowBits
  private val bufferRows    = 4

  @public
  val io = IO(new Bundle {
    val ex_st_i = Flipped(Decoupled(new MatrixResultRow))

    val store_req_o       = Output(Bool())
    val store_ctrl_resp_i = Flipped(Decoupled(new MatrixStoreCtrlResp(b)))
    val store_done_o      = Output(Bool())

    val wr_o = Decoupled(new MatrixStoreWriteReq(b))
    val wr_done_i = Input(Bool())
  })

  private def rowElem(row: UInt, col: Int): UInt =
    row((col + 1) * accElemBits - 1, col * accElemBits)

  private def packRow(elems: Vec[UInt], validElems: UInt): UInt =
    Cat((0 until tile).reverse.map { col =>
      Mux(col.U < validElems, elems(col), 0.U(accElemBits.W))
    })

  val sIdle :: sWaitResp :: sProcess :: sWrite :: sWaitWrite :: sDone :: Nil = Enum(6)
  val state = RegInit(sIdle)

  val resultBuffer = Reg(Vec(bufferRows, UInt(resultRowBits.W)))
  val readPtr      = RegInit(0.U(log2Ceil(bufferRows).W))
  val writePtr     = RegInit(0.U(log2Ceil(bufferRows).W))
  val bufferCount  = RegInit(0.U(log2Ceil(bufferRows + 1).W))

  io.ex_st_i.ready := bufferCount < bufferRows.U
  val bufferEnq = io.ex_st_i.fire
  val bufferDeq = state === sDone

  when(bufferEnq) {
    resultBuffer(writePtr) := io.ex_st_i.bits.data
    writePtr := Mux(writePtr === (bufferRows - 1).U, 0.U, writePtr + 1.U)
  }
  when(bufferDeq) {
    readPtr := Mux(readPtr === (bufferRows - 1).U, 0.U, readPtr + 1.U)
  }
  switch(Cat(bufferEnq, bufferDeq)) {
    is("b10".U) { bufferCount := bufferCount + 1.U }
    is("b01".U) { bufferCount := bufferCount - 1.U }
  }

  val respReg = RegInit(0.U.asTypeOf(new MatrixStoreCtrlResp(b)))

  io.store_req_o := state === sIdle && bufferCount =/= 0.U
  io.store_ctrl_resp_i.ready := state === sWaitResp
  io.store_done_o := state === sDone

  io.wr_o.valid := state === sWrite

  val currentRow = resultBuffer(readPtr)
  val rowElems = Wire(Vec(tile, UInt(accElemBits.W)))
  for (col <- 0 until tile) {
    rowElems(col) := rowElem(currentRow, col)
  }

  io.wr_o.bits.wr_bank       := respReg.wr_bank
  io.wr_o.bits.wr_group_base := respReg.wr_group_base
  io.wr_o.bits.wr_row_addr   := respReg.wr_row_addr
  io.wr_o.bits.valid_elems   := respReg.row_valid_elems
  io.wr_o.bits.beat_count    := (respReg.row_valid_elems + 3.U) >> 2
  io.wr_o.bits.data          := packRow(rowElems, respReg.row_valid_elems)

  when(state === sIdle && bufferCount =/= 0.U) {
    state := sWaitResp
  }

  when(io.store_ctrl_resp_i.fire) {
    respReg := io.store_ctrl_resp_i.bits
    state := sProcess
  }

  when(state === sProcess) {
    assert(bufferCount =/= 0.U, "MatrixStore: processing without a result row")

    when(respReg.row_write_valid) {
      state := sWrite
    }.otherwise {
      state := sDone
    }
  }

  when(io.wr_o.fire) {
    state := sWaitWrite
  }

  when(state === sWaitWrite && io.wr_done_i) {
    state := sDone
  }

  when(state === sDone) {
    state := sIdle
  }

  assert(!(io.store_req_o && state =/= sIdle), "MatrixStore: store_req must only pulse from idle")
  assert(io.wr_o.bits.valid_elems <= tile.U, "MatrixStore: row valid element count exceeds TILE")
  assert(bufferCount <= bufferRows.U, "MatrixStore: result buffer overflow")
}
