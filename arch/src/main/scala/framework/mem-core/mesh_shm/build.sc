import mill._
import mill.scalalib._

val frameworkRoot = os.pwd / os.up / os.up
val archRoot = frameworkRoot / os.up / os.up / os.up / os.up

trait ChiselModule extends SbtModule {
  def moduleRoot: os.Path

  override def millSourcePath = moduleRoot
  override def scalaVersion = "2.13.16"
  override def ivyDeps = Agg(ivy"org.chipsalliance::chisel:6.7.0")
  override def scalacPluginIvyDeps = Agg(ivy"org.chipsalliance:::chisel-plugin:6.7.0")
  override def scalacOptions = Seq("-deprecation", "-feature", "-language:reflectiveCalls", "-Ymacro-annotations")
}

object axis extends ChiselModule {
  override def moduleRoot = os.pwd / os.up / "axis"
}

object bank extends ChiselModule {
  override def moduleRoot = os.pwd / os.up / "bank"
  override def moduleDeps = Seq(axis)
}

object cde extends ChiselModule {
  override def moduleRoot = archRoot / "thirdparty" / "chipyard" / "tools" / "cde"
  override def sources = T.sources {
    super.sources() ++ Seq(PathRef(moduleRoot / "cde" / "src" / "chipsalliance"))
  }
}

object global_config extends ChiselModule {
  override def moduleRoot = frameworkRoot / "top"
  override def sources = T.sources {
    Seq(
      frameworkRoot / "top" / "GlobalConfig.scala",
      frameworkRoot / "top" / "configs" / "SimParam.scala",
      frameworkRoot / "memdomain" / "configs" / "MemDomainParam.scala",
      frameworkRoot / "frontend" / "configs" / "FrontendParam.scala",
      frameworkRoot / "gpdomain" / "configs" / "GpDomainParam.scala",
      frameworkRoot / "balldomain" / "configs" / "BallDomainParam.scala",
      frameworkRoot / "system" / "tile" / "configs" / "TileParam.scala"
    ).map(PathRef(_))
  }
  override def moduleDeps = Seq(cde)
  override def ivyDeps = super.ivyDeps() ++ Agg(ivy"com.lihaoyi::upickle:3.3.1")
}

object mesh_shm extends ChiselModule {
  override def moduleRoot = os.pwd
  override def moduleDeps = Seq(bank, global_config)
}
