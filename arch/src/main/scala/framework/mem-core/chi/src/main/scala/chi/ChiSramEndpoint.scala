package memcore.bus.chi

import chisel3._
import chisel3.util._

class ChiSnPort(p: ChiParams) extends Bundle {
  val rxReq           = Flipped(new ChiChannel((new ChiReq(p)).flitWidth))
  val rxDat           = Flipped(new ChiChannel((new ChiDat(p)).flitWidth))
  val txRsp           = new ChiChannel((new ChiRsp(p)).flitWidth)
  val txDat           = new ChiChannel((new ChiDat(p)).flitWidth)
  val rxLinkActiveReq = Input(Bool())
  val rxLinkActiveAck = Output(Bool())
  val txLinkActiveReq = Output(Bool())
  val txLinkActiveAck = Input(Bool())
  val rxSActive       = Input(Bool())
  val txSActive       = Output(Bool())
}

// Always-on, single-clock CHI SN-side SRAM endpoint. Activate after reset;
// retain both links until coordinated reset. No runtime link deactivation.
class ChiSramEndpoint(
  p:       ChiParams = ChiParams(),
  nodeId:  Int = 1,
  slots:   Int = 8,
  lines:   Int = 256,
  rxDepth: Int = 4)
    extends Module {

  val io = IO(new Bundle {
    val chi         = new ChiSnPort(p)
    val outstanding = Output(UInt(log2Ceil(slots + 1).W))
  })

  val started  = RegNext(!reset.asBool, false.B)
  val rxAck    = RegNext(io.chi.rxLinkActiveReq && started, false.B)
  val txActive = RegNext(io.chi.txLinkActiveAck && started, false.B)
  io.chi.txLinkActiveReq := started
  io.chi.rxLinkActiveAck := rxAck
  io.chi.txSActive       := started
  when(rxAck)(assert(io.chi.rxLinkActiveReq, "Runtime RX link deactivation is unsupported"))

  val rxReq = Module(new ChiRx((new ChiReq(p)).flitWidth, rxDepth))
  val rxDat = Module(new ChiRx((new ChiDat(p)).flitWidth, rxDepth))
  val txRsp = Module(new ChiTx((new ChiRsp(p)).flitWidth))
  val txDat = Module(new ChiTx((new ChiDat(p)).flitWidth))
  rxReq.io.active := rxAck
  rxDat.io.active := rxAck
  txRsp.io.active := txActive
  txDat.io.active := txActive
  rxReq.io.link <> io.chi.rxReq
  rxDat.io.link <> io.chi.rxDat
  io.chi.txRsp <> txRsp.io.link
  io.chi.txDat <> txDat.io.link

  val node   = Module(new ChiMemoryNode(p, nodeId, slots))
  val memory = Module(new ChiLineSram(p, lines))
  node.io.req.bits.unpack(rxReq.io.out.bits)
  node.io.req.valid   := rxReq.io.out.valid
  rxReq.io.out.ready  := node.io.req.ready
  node.io.rxDat.bits.unpack(rxDat.io.out.bits)
  node.io.rxDat.valid := rxDat.io.out.valid
  rxDat.io.out.ready  := node.io.rxDat.ready
  txRsp.io.in.bits    := node.io.rsp.bits.packed
  txRsp.io.in.valid   := node.io.rsp.valid
  node.io.rsp.ready   := txRsp.io.in.ready
  txDat.io.in.bits    := node.io.txDat.bits.packed
  txDat.io.in.valid   := node.io.txDat.valid
  node.io.txDat.ready := txDat.io.in.ready
  memory.io.req <> node.io.memoryReq
  node.io.memoryResp <> memory.io.resp
  io.outstanding      := node.io.outstanding
}

object EmitChiSram extends App {
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new ChiSramEndpoint(),
    firtoolOpts = args.drop(1) ++ Seq("--split-verilog", "-o=build"),
    args = Array("--target-dir", "build")
  )
}
