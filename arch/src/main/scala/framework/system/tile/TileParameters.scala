package framework.system.tile

import chisel3.util.log2Ceil
import framework.system.core.rocket.CpuParams
import framework.system.core.rocket.configs.RocketCpuParam

object TileParameters {

  def cpuParameters(
    cpu:                 RocketCpuParam,
    t:                   framework.system.tile.configs.TileParam,
    maxHartId:           Int,
    physicalAddressBits: Int
  ): CpuParams = {
    require(physicalAddressBits > t.pgIdxBits && physicalAddressBits <= t.paddrBits)
    require(
      !cpu.useVM || physicalAddressBits < t.vaddrBits,
      "Rocket requires platform physical addresses narrower than its virtual address format"
    )
    val core = RocketCpuParam.toRocketCoreParams(cpu, t.xLen, t.pgLevels).copy(
      useUser = cpu.useVM,
      useSupervisor = cpu.useVM,
      useHypervisor = false,
      useDebug = false,
      nPMPs = t.nPMPs,
      haveCease = false,
      haveSimTimeout = false
    )
    CpuParams(
      core = core,
      dcache = Some(RocketCpuParam.toDCacheParams(cpu, t.coreDataBytes * 8, 64)),
      icache = Some(RocketCpuParam.toICacheParams(cpu, t.coreDataBytes * 8, 64)),
      btb = RocketCpuParam.toBTBParams(cpu),
      physicalAddressBits = physicalAddressBits,
      hartIdBits = math.max(1, log2Ceil(maxHartId + 1)),
      beatBytes = t.coreDataBytes
    )
  }

}
