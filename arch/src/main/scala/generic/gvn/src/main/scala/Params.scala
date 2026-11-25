package framework.generic.gvn

import chisel3.util.log2Ceil
import framework.generic.bus.axi4

/**
 * Checker on Chip parameters.
 *
 * @param buckets
 *   summary buckets B (power of two); one beat of the checked tensor adds to one bucket
 * @param entries
 *   TMT entries, one per subgraph fence of a stimulus
 * @param accBits
 *   bucket accumulator width; accumulation wraps, only the deviation must fit
 * @param devBits
 *   bucket deviation width fed to the statistics (saturated)
 * @param awDepth
 *   snooped AW queue depth
 * @param wDepth
 *   snooped W queue depth (W may run ahead of its AW)
 */
case class Params(
  buckets: Int,
  entries: Int,
  axi:     axi4.Params,
  accBits: Int = 48,
  devBits: Int = 16,
  awDepth: Int = 4,
  wDepth:  Int = 16) {
  require(buckets >= 2 && (buckets & (buckets - 1)) == 0 && buckets <= 4096)
  require(entries >= 1 && entries <= 1024)
  require(axi.dataBits == 128, "lanes are defined for 128-bit beats")
  val bucketBits = log2Ceil(buckets)
  val entryBits  = math.max(1, log2Ceil(entries))
  val s2Bits     = 2 * devBits + bucketBits
  val s4Bits     = 4 * devBits + bucketBits
}
