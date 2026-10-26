package framework.ip.memory.cache.configs

import buckyball.config.CacheHardware

import chisel3.util.{isPow2, log2Ceil}

case class CacheParams(
  addressBits:   Int,
  lineBytes:     Int,
  sets:          Int,
  ways:          Int,
  idBits:        Int,
  metadataBits:  Int,
  responseDepth: Int) {
  require(lineBytes >= 8 && isPow2(lineBytes))
  require(sets >= 2 && isPow2(sets))
  require(ways >= 1 && isPow2(ways))
  require(idBits > 0 && metadataBits > 0 && responseDepth >= 2)
  val offsetBits = log2Ceil(lineBytes)
  val setBits    = log2Ceil(sets)
  val wayBits    = math.max(1, log2Ceil(ways))
  val tagBits    = addressBits - offsetBits - setBits
  val lineBits   = lineBytes * 8
  require(tagBits > 0)
}

object CacheParams {

  def fromProto(p: CacheHardware): CacheParams = CacheParams(
    p.getAddressBits,
    p.getLineBytes,
    p.getSets,
    p.getWays,
    p.getIdBits,
    p.getMetadataBits,
    p.getResponseDepth
  )

}
