import mill._
import mill.scalalib._

val archRoot = os.pwd / os.up / os.up / os.up / os.up / os.up / os.up

object cde extends SbtModule {
  override def millSourcePath = archRoot / "thirdparty" / "chipyard" / "tools" / "cde"
  override def scalaVersion = "2.13.16"
  override def sources = T.sources {
    super.sources() ++ Seq(PathRef(millSourcePath / "cde" / "src" / "chipsalliance"))
  }
}

object hardfloat extends SbtModule {
  override def millSourcePath = archRoot / "thirdparty" / "berkeley-hardfloat"
  override def scalaVersion = "2.13.16"
  override def moduleDeps = Seq(cde)
  override def sources = T.sources {
    super.sources() ++ Seq(PathRef(millSourcePath / "hardfloat" / "src" / "main" / "scala"))
  }
  override def ivyDeps = Agg(ivy"org.chipsalliance::chisel:6.7.0")
  override def scalacPluginIvyDeps = Agg(ivy"org.chipsalliance:::chisel-plugin:6.7.0")
}

object rvv extends SbtModule {
  override def millSourcePath = os.pwd
  override def scalaVersion = "2.13.16"
  override def moduleDeps = Seq(hardfloat)
  override def ivyDeps = Agg(ivy"org.chipsalliance::chisel:6.7.0")
  override def scalacPluginIvyDeps = Agg(ivy"org.chipsalliance:::chisel-plugin:6.7.0")
  override def scalacOptions = Seq("-language:reflectiveCalls")
}
