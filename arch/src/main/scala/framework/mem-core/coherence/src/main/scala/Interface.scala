package memcore.memory.coherence

import chisel3._
import chisel3.util._
import memcore.bus.chi._
import memcore.bus.chi.snf.{LineRequest, LineResponse}
import memcore.memory.coherence.configs.CoherenceParams

class DirectedSnoop(p: Params) extends Bundle {
  val targetNode = UInt(p.nodeIdBits.W)
  val flit       = new SnoopFlit(p)
}

class CoherenceIO(p: CoherenceParams) extends Bundle {
  val req         = Flipped(Decoupled(new RequestFlit(p.chi)))
  val rxRsp       = Flipped(Decoupled(new ResponseFlit(p.chi)))
  val rxDat       = Flipped(Decoupled(new DataFlit(p.chi)))
  val rsp         = Decoupled(new ResponseFlit(p.chi))
  val dat         = Decoupled(new DataFlit(p.chi))
  val snp         = Decoupled(new DirectedSnoop(p.chi))
  val memoryReq   = Decoupled(new LineRequest(p.chi))
  val memoryResp  = Flipped(Decoupled(new LineResponse(p.chi)))
  val outstanding = Output(UInt(log2Ceil(p.mshrEntries + 1).W))
}
