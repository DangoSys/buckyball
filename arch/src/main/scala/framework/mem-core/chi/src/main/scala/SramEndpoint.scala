package memcore.bus.chi

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}

class SnPort(p: Params) extends Bundle {
  val rxReq           = Flipped(new Channel((new RequestFlit(p)).flitWidth))
  val rxDat           = Flipped(new Channel((new DataFlit(p)).flitWidth))
  val txRsp           = new Channel((new ResponseFlit(p)).flitWidth)
  val txDat           = new Channel((new DataFlit(p)).flitWidth)
  val rxLinkActiveReq = Input(Bool())
  val rxLinkActiveAck = Output(Bool())
  val txLinkActiveReq = Output(Bool())
  val txLinkActiveAck = Input(Bool())
  val rxSActive       = Input(Bool())
  val txSActive       = Output(Bool())
}

class SramEndpointIO(p: Params, slots: Int) extends Bundle {
  val chi         = new SnPort(p)
  val outstanding = Output(UInt(log2Ceil(slots + 1).W))
}

// Always-on, single-clock CHI SN-side SRAM endpoint. Activate after reset;
// retain both links until coordinated reset. No runtime link deactivation.
@instantiable
class SramEndpoint(
  p:       Params = Params(),
  nodeId:  Int = 1,
  slots:   Int = 8,
  lines:   Int = 256,
  rxDepth: Int = 4)
    extends Module {

  @public
  val io = IO(new SramEndpointIO(p, slots))

  val started  = RegNext(!reset.asBool, false.B)
  val rxAck    = RegNext(io.chi.rxLinkActiveReq && started, false.B)
  val txActive = RegNext(io.chi.txLinkActiveAck && started, false.B)
  io.chi.txLinkActiveReq := started
  io.chi.rxLinkActiveAck := rxAck
  io.chi.txSActive       := started
  when(rxAck)(assert(io.chi.rxLinkActiveReq, "Runtime RX link deactivation is unsupported"))

  val rxReq: Instance[Rx] = Instantiate(new Rx((new RequestFlit(p)).flitWidth, rxDepth))
  val rxDat: Instance[Rx] = Instantiate(new Rx((new DataFlit(p)).flitWidth, rxDepth))
  val txRsp: Instance[Tx] = Instantiate(new Tx((new ResponseFlit(p)).flitWidth))
  val txDat: Instance[Tx] = Instantiate(new Tx((new DataFlit(p)).flitWidth))
  rxReq.io.active := rxAck
  rxDat.io.active := rxAck
  txRsp.io.active := txActive
  txDat.io.active := txActive
  rxReq.io.link <> io.chi.rxReq
  rxDat.io.link <> io.chi.rxDat
  io.chi.txRsp <> txRsp.io.link
  io.chi.txDat <> txDat.io.link

  val node:   Instance[MemoryNode] = Instantiate(new MemoryNode(p, nodeId, slots))
  val memory: Instance[LineSram]   = Instantiate(new LineSram(p, lines))
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
