package framework.system.core

import freechips.rocketchip.rocket.{BTBParams, DCacheParams, ICacheParams}
import framework.system.sm.SMParams
import framework.system.core.scu.SCUParams

/**
 * BBCore 参数
 *
 * 一个 Core = N 个 SM + 每 SM 独立 L1 + 每 SM 独立 PTW + SCU
 *
 * @param nSMs 这个 Core 有几个 SM
 * @param smParams 所有 SM 使用相同配置
 * @param l1ICache L1 ICache 参数 (每个 SM 一个独立的)
 * @param l1DCache L1 DCache 参数 (每个 SM 一个独立的)
 * @param btb 可选的 BTB 参数
 * @param scu SCU 参数
 * @param beuAddr 可选的 Bus Error Unit 地址
 */
case class BBCoreParams(
  nSMs:     Int,
  smParams: SMParams,
  l1ICache: ICacheParams,
  l1DCache: DCacheParams,
  btb:      Option[BTBParams] = None,
  scu:      SCUParams = SCUParams(),
  beuAddr:  Option[BigInt] = None) {
  require(nSMs >= 1, "Core must have at least 1 SM")

  /** 这个 Core 内有多少个 SM 挂了 Buckyball */
  val nBuckyballSMs: Int = if (smParams.hasBuckyball) nSMs else 0

  /** 这个 Core 总 hartid 数 = SM 数 */
  val totalHarts: Int = nSMs
}
