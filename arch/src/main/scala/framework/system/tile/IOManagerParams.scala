package framework.system.tile

/**
 * MMIO 设备参数
 */
case class MMIODeviceParams(
  name:     String,
  baseAddr: BigInt,
  size:     BigInt)

/**
 * IOManager 参数
 *
 * IOManager 负责汇聚所有 Core 的 TileLink 流量。
 *
 * @param nCores 管理几个 Core
 * @param enableQoS 是否启用 QoS (暂时未实现)
 * @param mmioDevices MMIO 设备列表
 * @param tlAddrBits TileLink 地址位宽
 * @param tlDataBits TileLink 数据位宽
 * @param tlSourceBits TileLink source ID 位宽 (default: 4 = 16 sources, enough for small configs)
 */
case class IOManagerParams(
  nCores:       Int,
  enableQoS:    Boolean = false,
  mmioDevices:  Seq[MMIODeviceParams] = Nil,
  tlAddrBits:   Int = 32,
  tlDataBits:   Int = 64,
  tlSourceBits: Int = 4)
