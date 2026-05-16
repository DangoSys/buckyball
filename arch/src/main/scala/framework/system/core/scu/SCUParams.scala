package framework.system.core.scu

/**
 * SCU (Synchronization Control Unit) 参数
 *
 * SCU 在 Core 内部拦截 MMIO 流量,并提供同步原语。
 *
 * @param mmioBase MMIO 拦截的基地址
 * @param mmioSize MMIO 拦截的地址范围大小
 * @param interceptMMIO 是否启用 MMIO 拦截
 */
case class SCUParams(
  mmioBase:      BigInt = 0x10000000L,
  mmioSize:      BigInt = 0x10000000L,
  interceptMMIO: Boolean = true)
