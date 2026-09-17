import mill._
import mill.scalalib._

trait ChiselModule extends SbtModule {
  def moduleRoot: os.Path

  override def millSourcePath = moduleRoot
  override def scalaVersion = "2.13.16"
  override def ivyDeps = Agg(ivy"org.chipsalliance::chisel:6.7.0")
  override def scalacPluginIvyDeps = Agg(ivy"org.chipsalliance:::chisel-plugin:6.7.0")
  override def scalacOptions = Seq("-deprecation", "-feature", "-language:reflectiveCalls")
}

object chi extends ChiselModule {
  override def moduleRoot = os.pwd / os.up / "chi"
}

object cache extends ChiselModule {
  override def moduleRoot = os.pwd
  override def moduleDeps = Seq(chi)
}
