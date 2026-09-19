package memcore.bus.chi

import chisel3._
import chisel3.util._

// The internal protocol boundary; physical CHI channels are attached by the fabric.
class RequesterPort(p: Params) extends Bundle {
  val req   = Decoupled(new RequestFlit(p))
  val txRsp = Decoupled(new ResponseFlit(p))
  val txDat = Decoupled(new DataFlit(p))
  val snp   = Flipped(Decoupled(new SnoopFlit(p)))
  val rxRsp = Flipped(Decoupled(new ResponseFlit(p)))
  val rxDat = Flipped(Decoupled(new DataFlit(p)))
}
