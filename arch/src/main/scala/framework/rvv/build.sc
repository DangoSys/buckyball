import mill._
import mill.scalalib._

val archRoot = os.pwd / os.up / os.up / os.up / os.up / os.up

object cde extends SbtModule {
  override def millSourcePath = archRoot / "thirdparty" / "rocket-chip" / "dependencies" / "cde"
  override def scalaVersion   = "2.13.16"

  override def sources = T.sources {
    super.sources() ++ Seq(PathRef(millSourcePath / "cde" / "src" / "chipsalliance"))
  }

}

object hardfloat extends SbtModule {
  override def millSourcePath = archRoot / "thirdparty" / "berkeley-hardfloat"
  override def scalaVersion   = "2.13.16"
  override def moduleDeps     = Seq(cde)

  override def sources = T.sources {
    super.sources() ++ Seq(PathRef(millSourcePath / "hardfloat" / "src" / "main" / "scala"))
  }

  override def ivyDeps             = Agg(ivy"org.chipsalliance::chisel:6.7.0")
  override def scalacPluginIvyDeps = Agg(ivy"org.chipsalliance:::chisel-plugin:6.7.0")
}

object rvv extends SbtModule {
  override def millSourcePath = os.pwd
  override def scalaVersion   = "2.13.16"
  override def moduleDeps     = Seq(hardfloat)

  override def sources = T.sources {
    super.sources() ++ Seq(
      PathRef(archRoot / "src" / "main" / "scala" / "framework" / "top" / "GlobalConfig.scala"),
      PathRef(archRoot / "src" / "main" / "scala" / "framework" / "top" / "configs" / "SimParam.scala"),
      PathRef(archRoot / "src" / "main" / "scala" / "framework" / "memdomain" / "configs" / "MemDomainParam.scala"),
      PathRef(archRoot / "src" / "main" / "scala" / "framework" / "frontend" / "configs" / "FrontendParam.scala"),
      PathRef(archRoot / "src" / "main" / "scala" / "framework" / "balldomain" / "configs" / "BallDomainParam.scala"),
      PathRef(archRoot / "src" / "main" / "scala" / "framework" / "system" / "tile" / "configs" / "TileParam.scala"),
      PathRef(archRoot / "src" / "main" / "scala" / "framework" / "balldomain" / "blink" / "blink.scala"),
      PathRef(archRoot / "src" / "main" / "scala" / "framework" / "balldomain" / "blink" / "bank.scala"),
      PathRef(archRoot / "src" / "main" / "scala" / "framework" / "balldomain" / "blink" / "status.scala"),
      PathRef(archRoot / "src" / "main" / "scala" / "framework" / "balldomain" / "blink" / "baseball.scala"),
      PathRef(archRoot / "src" / "main" / "scala" / "framework" / "balldomain" / "blink" / "SubRobRow.scala"),
      PathRef(archRoot / "src" / "main" / "scala" / "framework" / "balldomain" / "blink" / "mmio" / "MmioRead.scala"),
      PathRef(archRoot / "src" / "main" / "scala" / "framework" / "balldomain" / "blink" / "mmio" / "MmioWrite.scala"),
      PathRef(archRoot / "src" / "main" / "scala" / "framework" / "memdomain" / "backend" / "banks" / "SramIO.scala"),
      PathRef(archRoot / "src" / "main" / "scala" / "framework" / "memdomain" / "backend" / "mmio" / "MmioIO.scala"),
      PathRef(archRoot / "src" / "main" / "scala" / "framework" / "balldomain" / "rs" / "interfaces.scala"),
      PathRef(archRoot / "src" / "main" / "scala" / "framework" / "balldomain" / "decoder" / "BallDecodeCmd.scala"),
      PathRef(archRoot / "src" / "main" / "scala" / "framework" / "frontend" / "decoder" / "PostGDCmd.scala"),
      PathRef(archRoot / "src" / "main" / "scala" / "framework" / "frontend" / "scoreboard" / "BankAccessInfo.scala"),
      PathRef(archRoot / "src" / "main" / "scala" / "framework" / "system" / "core" / "rocket" / "RoCCCommandBB.scala")
    )
  }

  override def ivyDeps             = Agg(ivy"org.chipsalliance::chisel:6.7.0", ivy"com.lihaoyi::upickle:3.3.1")
  override def scalacPluginIvyDeps = Agg(ivy"org.chipsalliance:::chisel-plugin:6.7.0")
  override def scalacOptions       = Seq("-language:reflectiveCalls", "-Ymacro-annotations")
  override def mainClass           = Some("framework.rvv.Emit")
}
