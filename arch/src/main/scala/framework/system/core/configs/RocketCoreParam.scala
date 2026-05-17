package framework.system.core.configs

import upickle.default._
import freechips.rocketchip.rocket.{BTBParams, DCacheParams, ICacheParams, MulDivParams, RocketCoreParams}
import freechips.rocketchip.tile.FPUParams

/**
 * JSON-serializable Rocket Core configuration parameters.
 *
 * Phase 1: Common parameters (xLen, pgLevels, useVM, Zb*, mulDiv, fpu, cache, btb)
 * Phase 2: Advanced parameters (TLB, ECC, clockGate, etc.) - to be added later
 *
 * Optional features (mulDiv, fpu, btb) use an `enable` flag instead of Option[T],
 * so JSON stays as a plain dictionary (upickle serializes Option as array).
 */

case class MulDivParam(
  enable:      Boolean = true,
  mulUnroll:   Int = 8,
  mulEarlyOut: Boolean = true,
  divEarlyOut: Boolean = true)

object MulDivParam {
  implicit val rw: ReadWriter[MulDivParam] = macroRW

  def toMulDivParams(p: MulDivParam): Option[MulDivParams] =
    if (p.enable) Some(MulDivParams(
      mulUnroll = p.mulUnroll,
      mulEarlyOut = p.mulEarlyOut,
      divEarlyOut = p.divEarlyOut
    ))
    else None

}

case class FPUParam(
  enable:  Boolean = true,
  minFLen: Int = 16,
  fLen:    Int = 64)

object FPUParam {
  implicit val rw: ReadWriter[FPUParam] = macroRW

  def toFPUParams(p: FPUParam): Option[FPUParams] =
    if (p.enable) Some(FPUParams(
      minFLen = p.minFLen,
      fLen = p.fLen
    ))
    else None

}

case class DCacheParam(
  nSets:  Int = 64,
  nWays:  Int = 8,
  nMSHRs: Int = 0)

object DCacheParam {
  implicit val rw: ReadWriter[DCacheParam] = macroRW

  def toDCacheParams(p: DCacheParam, rowBits: Int, blockBytes: Int): DCacheParams = DCacheParams(
    nSets = p.nSets,
    nWays = p.nWays,
    rowBits = rowBits,
    nMSHRs = p.nMSHRs,
    blockBytes = blockBytes
  )

}

case class ICacheParam(
  nSets: Int = 64,
  nWays: Int = 8)

object ICacheParam {
  implicit val rw: ReadWriter[ICacheParam] = macroRW

  def toICacheParams(p: ICacheParam, rowBits: Int, blockBytes: Int): ICacheParams = ICacheParams(
    nSets = p.nSets,
    nWays = p.nWays,
    rowBits = rowBits,
    blockBytes = blockBytes
  )

}

case class BTBParam(
  enable:   Boolean = true,
  nEntries: Int = 28,
  nRAS:     Int = 6)

object BTBParam {
  implicit val rw: ReadWriter[BTBParam] = macroRW

  def toBTBParams(p: BTBParam): Option[BTBParams] =
    if (p.enable) Some(BTBParams(
      nEntries = p.nEntries,
      nRAS = p.nRAS
    ))
    else None

}

case class RocketCoreParam(
  xLen:     Int = 64,
  pgLevels: Int = 3,
  useVM:    Boolean = true,
  useZba:   Boolean = true,
  useZbb:   Boolean = true,
  useZbs:   Boolean = true,
  mulDiv:   MulDivParam = MulDivParam(),
  fpu:      FPUParam = FPUParam(),
  dcache:   DCacheParam = DCacheParam(),
  icache:   ICacheParam = ICacheParam(),
  btb:      BTBParam = BTBParam())

object RocketCoreParam {
  implicit val rw: ReadWriter[RocketCoreParam] = macroRW

  /**
   * Convert to rocket-chip RocketCoreParams.
   * rowBits and blockBytes are injected from site(SystemBusKey) and site(CacheBlockBytes).
   */
  def toRocketCoreParams(p: RocketCoreParam): RocketCoreParams = RocketCoreParams(
    xLen = p.xLen,
    pgLevels = p.pgLevels,
    useVM = p.useVM,
    useZba = p.useZba,
    useZbb = p.useZbb,
    useZbs = p.useZbs,
    mulDiv = MulDivParam.toMulDivParams(p.mulDiv),
    fpu = FPUParam.toFPUParams(p.fpu)
  )

  def toDCacheParams(p: RocketCoreParam, rowBits: Int, blockBytes: Int): DCacheParams =
    DCacheParam.toDCacheParams(p.dcache, rowBits, blockBytes)

  def toICacheParams(p: RocketCoreParam, rowBits: Int, blockBytes: Int): ICacheParams =
    ICacheParam.toICacheParams(p.icache, rowBits, blockBytes)

  def toBTBParams(p: RocketCoreParam): Option[BTBParams] =
    BTBParam.toBTBParams(p.btb)
}
