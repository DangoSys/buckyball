import mill._
import mill.scalalib._

object queue extends SbtModule {
  override def millSourcePath      = os.pwd / os.up / "queue"
  override def scalaVersion        = "2.13.16"
  override def ivyDeps             = Agg(ivy"org.chipsalliance::chisel:6.7.0")
  override def scalacPluginIvyDeps = Agg(ivy"org.chipsalliance:::chisel-plugin:6.7.0")
  override def scalacOptions       = Seq("-language:reflectiveCalls", "-Ymacro-annotations")
}

object axi4 extends SbtModule {
  override def moduleDeps          = Seq(queue)
  override def millSourcePath      = os.pwd
  override def scalaVersion        = "2.13.16"
  override def ivyDeps             = Agg(ivy"org.chipsalliance::chisel:6.7.0")
  override def scalacPluginIvyDeps = Agg(ivy"org.chipsalliance:::chisel-plugin:6.7.0")
  override def scalacOptions       = Seq("-deprecation", "-feature", "-language:reflectiveCalls", "-Ymacro-annotations")
  override def mainClass           = Some("memcore.bus.axi4.Emit")
}
