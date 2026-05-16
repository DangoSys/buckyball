package framework.system.sm

import freechips.rocketchip.rocket.RocketCoreParams
import framework.top.GlobalConfig

/**
 * SM (Streaming Multiprocessor) 参数
 *
 * 一个 SM = 一个 Rocket 核心 + 可选的 Buckyball 加速器
 *
 * @param rocket Rocket 核心参数
 * @param globalConfig 全局基础配置 (始终存在, 提供 xLen/paddrBits 等基础参数)
 * @param hasBuckyball 是否启用 Buckyball 加速器 (NPU 是可选的, 但 GlobalConfig 始终存在)
 * @param enableClockGate 是否启用时钟门控
 */
case class SMParams(
  rocket:          RocketCoreParams,
  globalConfig:    GlobalConfig,
  hasBuckyball:    Boolean = false,
  enableClockGate: Boolean = false) {

  /** 当 hasBuckyball=true 时, 用于 BuckyballAccelerator 的配置 */
  def buckyballConfig: GlobalConfig = globalConfig

  /** xLen 从 globalConfig 获取 */
  def xLen: Int = globalConfig.core.xLen
}
