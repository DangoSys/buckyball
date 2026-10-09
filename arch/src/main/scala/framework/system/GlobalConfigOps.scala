package framework.system

import framework.top.GlobalConfig
import framework.top.configs._
import framework.system.configloader._
import framework.system.core.{AntCoreParams, RocketCoreParams}
import framework.system.tile.{TileParameters, TileParams}
import framework.system.device._

/** Pure configuration projection and IP-parameter derivation; no hardware is instantiated here. */
object GlobalConfigOps {
  def accelerator(b: GlobalConfig):      AcceleratorConfig =
    AcceleratorConfig(b.memDomain, b.frontend, b.rvv, b.ballDomain, b.tile, b.sim, b.coreSignature)
  def accelerator(a: AcceleratorConfig): GlobalConfig      =
    GlobalConfig(a.memDomain, a.frontend, a.rvv, a.ballDomain, a.tile, a.sim, a.coreSignature)

  def core(name: String, c: TileCore): CoreDesign = c match {
    case RocketTileCore(cpu, b) => CoreDesign(name, Some(cpu), None, b.map(accelerator))
    case AntTileCore(local, b)  => CoreDesign(name, None, Some(local), Some(accelerator(b)))
    case _                      => throw new IllegalArgumentException("Unsupported core design")
  }

  def core(c: CoreDesign): TileCore = c.rocket match {
    case Some(cpu) => RocketTileCore(cpu, c.accelerator.map(accelerator))
    case None      => AntTileCore(c.local.get, accelerator(c.accelerator.get))
  }

  def shared(s: SharedConfig): SharedStorageParams = SharedStorageParams(s.bankBits, s.bankEntries, s.banks)

  def design(t: TileTopology): TileDesign = {
    require(t.coreFactories.size == t.cores.size, "Every core requires a configured design")
    TileDesign(
      t.factory,
      t.param,
      t.main,
      t.coreFactories.zip(t.cores).map { case (n, c) => core(n, c) },
      t.signatures,
      t.controller,
      t.sharedStorage.map(s => SharedConfig(s.bankBits, s.bankEntries, s.banks))
    )
  }

  def fromSystem(p: SystemParams): GlobalConfig = {
    val d        = p.devices
    val devices  = DevicesConfig(
      d.clint.base,
      d.clint.bytes,
      d.clint.tickCycles,
      d.plic.base,
      d.plic.bytes,
      d.plic.sources,
      d.plic.priorityBits,
      d.bootrom.map(r => RomConfig(r.base, r.bytes, r.image)),
      d.scu.baseAddress,
      d.scu.strideBytes,
      d.scu.totalSizeBytes,
      d.scu.maxHarts
    )
    val platform = PlatformConfig(p.cpuPhysicalBits, p.memory, p.l1, p.regions, p.tracking, p.ddr, devices)
    GlobalConfig().copy(systemDesign =
      Some(SystemDesign(
        p.topology.tiles.map(t => framework.top.configs.TilePlacement(design(t), t.hartIds, t.executionIds)),
        platform
      ))
    )
  }

  implicit class DesignParameters(val b: GlobalConfig) extends AnyVal {

    def systemParams: SystemParams = {
      val s        = b.systemDesign.get
      val p        = s.platform
      val d        = p.devices
      val topology = ExampleTopology(s.tiles.map { t =>
        val v = t.design
        TileTopology(
          v.param,
          v.main,
          v.cores.map(core),
          t.hartIds,
          t.executionIds,
          v.signatures,
          v.controller,
          v.sharedStorage.map(shared),
          v.cores.map(_.factory),
          v.factory
        )
      })
      val devices  = DeviceParams(
        ClintParams(d.clintBase, d.clintBytes, d.tickCycles),
        PlicParams(d.plicBase, d.plicBytes, d.sources, d.priorityBits),
        d.bootrom.map(r => BootRomParams(r.base, r.bytes, r.image)),
        sims.scu.SCUParams(d.scuBase, d.scuStride, d.scuBytes, d.scuMaxHarts)
      )
      SystemParams(topology, p.cpuPhysicalBits, p.memory, p.l1, p.regions, p.tracking, p.ddr, devices)
    }

    def forTile(index: Int): GlobalConfig = {
      val s   = b.systemDesign.get
      val ids = s.tiles.flatMap(_.hartIds)
      GlobalConfig().copy(tileDesign = Some(TileScope(s.tiles(index).design, s.platform, ids.max, ids.size, s.tiles.size)))
    }

    def tileLinkParams: framework.system.tile.TileLinkParams = {
      val t = b.tileDesign.get
      framework.system.tile.TileLinkParams(
        t.design.main,
        t.design.cores.count(_.rocket.isDefined),
        t.design.cores.size,
        if (t.design.main) t.controls else 0,
        t.platform.memory.chi,
        t.platform.ddr.axi,
        t.design.sharedStorage.isDefined
      )
    }

    def tileParams: TileParams = {
      val t    = b.tileDesign.get
      val p    = t.platform
      val cpus = t.design.cores.flatMap(_.rocket).map(cpu =>
        TileParameters.cpuParameters(cpu, t.design.param, t.maxHartId, p.cpuPhysicalBits)
      )
      TileParams(
        t.design.main,
        t.design.cores.map(core),
        t.design.signatures,
        t.design.controller,
        cpus,
        p.memory.copy(agents = 2 * cpus.size),
        p.l1,
        p.regions,
        p.tracking,
        t.controls,
        p.ddr.axi,
        t.design.sharedStorage.map(shared),
        t.tiles
      )
    }

    def forCore(index: Int): GlobalConfig = {
      val t         = b.tileDesign.get
      val c         = t.design.cores(index)
      val cpus      = t.design.cores.flatMap(_.rocket)
      val selected  = c.accelerator.map(accelerator).getOrElse(GlobalConfig())
      val signature = if (c.local.isDefined) selected.coreSignature else BigInt(0)
      selected.copy(coreDesign = Some(CoreScope(
        c.factory,
        c.rocket,
        c.local,
        c.accelerator.isDefined,
        t.design.controller.map(i => t.design.cores(i).rocket.get),
        t.design.param,
        t.platform,
        t.maxHartId,
        if (c.rocket.isDefined) t.design.cores.take(index).count(_.rocket.isDefined) else 0,
        cpus.size,
        signature
      )))
    }

    def rocketParams: RocketCoreParams = {
      val c   = b.coreDesign.get
      require(c.rocket.isDefined)
      val p   = c.platform
      val cpu = TileParameters.cpuParameters(c.rocket.get, c.tile, c.maxHartId, p.cpuPhysicalBits)
      RocketCoreParams(
        cpu,
        p.l1.copy(nodeId = c.cpuIndex + 1),
        p.l1.copy(nodeId = c.cpuCount + c.cpuIndex + 1),
        p.regions,
        hier.core.rocket.Commands(c.hasAccelerator, true, p.tracking),
        Option.when(c.hasAccelerator)(b)
      )
    }

    def antParams: AntCoreParams = {
      val c   = b.coreDesign.get
      require(c.local.isDefined && c.hasAccelerator)
      val p   = c.platform
      val cpu = TileParameters.cpuParameters(c.controller.get, c.tile, c.maxHartId, p.cpuPhysicalBits)
      AntCoreParams(b, c.local.get, p.tracking, p.memory.chi, c.signature, cpu)
    }

  }

}
