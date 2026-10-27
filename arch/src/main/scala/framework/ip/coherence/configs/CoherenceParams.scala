package framework.ip.memory.coherence.configs

import buckyball.config.HomeHardware

import chisel3.util.log2Ceil
import framework.ip.bus.chi.Params
import framework.ip.memory.cache.configs.CacheParams

case class CoherenceParams(
  chi:         Params,
  cache:       CacheParams,
  agents:      Int,
  mshrEntries: Int,
  homeId:      Int) {
  require(cache.lineBytes == 64 && cache.addressBits == chi.addressBits)
  require(agents >= 1 && agents < homeId && BigInt(homeId) < (BigInt(1) << chi.nodeIdBits))
  require(mshrEntries >= 1 && BigInt(mshrEntries) <= (BigInt(1) << chi.txnIdBits))
  require(BigInt(mshrEntries) <= (BigInt(1) << cache.idBits))
  val slotBits        = math.max(1, log2Ceil(mshrEntries))
  val lineAddressBits = chi.addressBits - cache.offsetBits
}

object CoherenceParams {

  def fromProto(p: HomeHardware): CoherenceParams =
    CoherenceParams(
      Params.fromProto(p.getRequesters),
      CacheParams.fromProto(p.getCache),
      p.getAgents,
      p.getMshrEntries,
      p.getNodeId
    )

}
