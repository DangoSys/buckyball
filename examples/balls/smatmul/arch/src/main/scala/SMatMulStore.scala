package framework.balldomain.prototype.systolicarray

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.top.GlobalConfig

@instantiable
class SystolicArrayStore(val b: GlobalConfig) extends Module {
  private val tile                 = SystolicArrayConst.Tile
  private val resultRowBits        = SystolicArrayConst.ResultRowBits
  private val bufferRows           = 2
  private val descriptorRows       = 2
  private val descriptorCountWidth = log2Ceil(descriptorRows + 1)

  @public
  val io = IO(new Bundle {
    val ex_st_i = Flipped(Decoupled(new SystolicResultRow))

    // Ctrl 主动按结果行顺序预发写回描述符；Store 只在本地地址 FIFO 有空位时接收。
    val store_ctrl_resp_i = Flipped(Decoupled(new SystolicStoreCtrlResp(b)))
    val store_done_o      = Output(Bool())

    val wr_o      = Decoupled(new SystolicStoreWriteReq(b))
    val wr_done_i = Input(Bool())
  })

  val resultBuffer = Reg(Vec(bufferRows, UInt(resultRowBits.W)))
  val readPtr      = RegInit(0.U(log2Ceil(bufferRows).W))
  val writePtr     = RegInit(0.U(log2Ceil(bufferRows).W))
  val bufferCount  = RegInit(0.U(log2Ceil(bufferRows + 1).W))

  // 四路 Unit 已在接收某行后持有该行的完整数据；Store 因而在 wr_o.fire 时
  // 即可释放本行，继续把下一行送入 Unit，而不是等待 bank 响应。
  val bufferEnq = io.ex_st_i.fire

  when(bufferEnq) {
    resultBuffer(writePtr) := io.ex_st_i.bits.data
    writePtr               := Mux(writePtr === (bufferRows - 1).U, 0.U, writePtr + 1.U)
  }

  // 结果行和写回描述符分别以 FIFO 顺序保存。两者的队首总是同一逻辑行，
  // 因而 resultBuffer 的物理环形回绕不会改变数据与地址的配对关系。
  val descriptorBuffer =
    RegInit(
      VecInit(
        Seq.fill(descriptorRows)(0.U.asTypeOf(new SystolicStoreCtrlResp(b)))
      )
    )

  val descriptorReadPtr  = RegInit(0.U(log2Ceil(descriptorRows).W))
  val descriptorWritePtr = RegInit(0.U(log2Ceil(descriptorRows).W))
  val descriptorCount    = RegInit(0.U(descriptorCountWidth.W))
  val descriptorEnq      = io.store_ctrl_resp_i.fire
  val currentRow         = resultBuffer(readPtr)
  val currentDescriptor  = descriptorBuffer(descriptorReadPtr)
  val pairAvailable      = bufferCount =/= 0.U && descriptorCount =/= 0.U

  when(descriptorEnq) {
    descriptorBuffer(descriptorWritePtr) := io.store_ctrl_resp_i.bits
    descriptorWritePtr                   := Mux(
      descriptorWritePtr === (descriptorRows - 1).U,
      0.U,
      descriptorWritePtr + 1.U
    )
  }
  io.wr_o.bits.rob_id        := currentDescriptor.rob_id
  io.wr_o.bits.wr_bank       := currentDescriptor.wr_bank
  io.wr_o.bits.wr_group_base := currentDescriptor.wr_group_base
  io.wr_o.bits.wr_row_addr   := currentDescriptor.wr_row_addr
  io.wr_o.bits.valid_elems   := currentDescriptor.row_valid_elems
  // Unit 依据 valid_elems 生成 byte mask；被 mask 的尾列不会写入 bank，数据无需再
  // 在 Store 中重复清零。
  io.wr_o.bits.data          := currentRow

  // Ctrl 只提供真实结果行的描述符；结果和描述符队首同时存在时即可交给 Unit。
  io.wr_o.valid := pairAvailable
  val descriptorDeq = io.wr_o.fire
  val bufferDeq     = descriptorDeq

  io.store_done_o := io.wr_done_i

  when(bufferDeq) {
    readPtr := Mux(readPtr === (bufferRows - 1).U, 0.U, readPtr + 1.U)
  }
  switch(Cat(bufferEnq, bufferDeq)) {
    is("b10".U)(bufferCount := bufferCount + 1.U)
    is("b01".U)(bufferCount := bufferCount - 1.U)
  }
  io.ex_st_i.ready := bufferCount < bufferRows.U || bufferDeq

  when(descriptorDeq) {
    descriptorReadPtr := Mux(
      descriptorReadPtr === (descriptorRows - 1).U,
      0.U,
      descriptorReadPtr + 1.U
    )
  }
  switch(Cat(descriptorEnq, descriptorDeq)) {
    is("b10".U)(descriptorCount := descriptorCount + 1.U)
    is("b01".U)(descriptorCount := descriptorCount - 1.U)
  }

  // 地址描述符与结果数据独立预取：即使 EX 尚未交付任何结果，只要 FIFO 有空位
  // 就接收 Ctrl 的下一行地址。descriptorDeq 允许同周期“消费一项、补入一项”。
  io.store_ctrl_resp_i.ready := descriptorCount < descriptorRows.U || descriptorDeq

  when(io.wr_o.valid) {
    assert(
      io.wr_o.bits.valid_elems <= tile.U,
      "SystolicArrayStore: row valid element count exceeds TILE"
    )
  }
  assert(
    bufferCount <= bufferRows.U,
    "SystolicArrayStore: result buffer overflow"
  )
  assert(
    descriptorCount <= descriptorRows.U,
    "SystolicArrayStore: descriptor buffer overflow"
  )
  when(descriptorDeq) {
    assert(
      bufferCount =/= 0.U,
      "SystolicArrayStore: completed without a result row"
    )
    assert(
      descriptorCount =/= 0.U,
      "SystolicArrayStore: completed without a write descriptor"
    )
  }
}
