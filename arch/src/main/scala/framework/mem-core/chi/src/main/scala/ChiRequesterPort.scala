package memcore.bus.chi

import chisel3._
import chisel3.util._

// The internal protocol boundary; physical CHI channels are attached by the fabric.
class ChiRequesterPort(p: ChiParams) extends Bundle {
  val req   = Decoupled(new ChiReq(p))
  val txRsp = Decoupled(new ChiRsp(p))
  val txDat = Decoupled(new ChiDat(p))
  val snp   = Flipped(Decoupled(new ChiSnp(p)))
  val rxRsp = Flipped(Decoupled(new ChiRsp(p)))
  val rxDat = Flipped(Decoupled(new ChiDat(p)))
}
