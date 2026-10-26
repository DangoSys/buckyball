package framework.device.peripherals.scu.configs

import buckyball.config.ScuHardware
import scala.jdk.CollectionConverters._

import upickle.default.{macroRW, ReadWriter}

/**
 * SCU is a global multi-hart System Control Unit (see SystemControl). It
 * provides a per-hart sub-region for UART output and simulation exit. Each hart accesses its own sub-region via
 * `baseAddress + hartId * strideBytes`.
 *
 * Address layout (default values):
 *   baseAddress     = 0x60000000
 *   strideBytes     = 0x40000   (256 KiB per hart)
 *   totalSizeBytes  = 0x10000000 (256 MiB total, supports up to 1024 harts)
 *   maxHarts        = 64
 *
 * Within each hart's sub-region:
 *   +0x00000: simExit register (write triggers $finish with exit code)
 *   +0x20000: uartTx register  (write outputs a byte via DPI-C)
 *   +0x20004: uartRx register  (read pops a byte from host input)
 *   +0x20005: status register  (read-only, bit0 rx-valid, bit5/6 tx-ready)
 *   +0x20006: ready register   (read/write software-visible byte)
 *
 * @param baseAddress   physical base address (must be aligned to totalSizeBytes)
 * @param strideBytes   bytes per hart (must be power of two)
 * @param totalSizeBytes total address window (must be power of two,
 *                       >= maxHarts * strideBytes)
 * @param maxHarts      number of hart sub-regions. Accesses targeting
 *                      hartId >= maxHarts return an access error.
 */
case class SCUParams(
  baseAddress:    BigInt = BigInt("60000000", 16),
  strideBytes:    BigInt = BigInt("40000", 16),
  totalSizeBytes: BigInt = BigInt("10000000", 16),
  maxHarts:       Int = 64) {
  require(
    strideBytes > 0 && (strideBytes & (strideBytes - 1)) == 0,
    s"SCU strideBytes ($strideBytes) must be a positive power of two"
  )
  require(
    totalSizeBytes > 0 && (totalSizeBytes & (totalSizeBytes - 1)) == 0,
    s"SCU totalSizeBytes ($totalSizeBytes) must be a positive power of two"
  )
  require(maxHarts > 0, s"SCU maxHarts ($maxHarts) must be positive")
  require(
    BigInt(maxHarts) * strideBytes <= totalSizeBytes,
    s"SCU maxHarts ($maxHarts) * strideBytes ($strideBytes) = " +
      s"${BigInt(maxHarts) * strideBytes} exceeds totalSizeBytes ($totalSizeBytes)"
  )
}

object SCUParams {

  def fromProto(p: ScuHardware): SCUParams = SCUParams(
    BigInt(java.lang.Long.toUnsignedString(p.getBaseAddress)),
    BigInt(java.lang.Long.toUnsignedString(p.getStrideBytes)),
    BigInt(java.lang.Long.toUnsignedString(p.getTotalSizeBytes)),
    p.getMaxHarts
  )

  implicit val rw: ReadWriter[SCUParams] = macroRW
}
