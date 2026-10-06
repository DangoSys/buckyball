import mill._
import mill.scalalib._

trait ChiselModule extends SbtModule {
  def moduleRoot: os.Path
  override def millSourcePath      = moduleRoot
  override def scalaVersion        = "2.13.16"
  override def ivyDeps             = Agg(ivy"org.chipsalliance::chisel:6.7.0")
  override def scalacPluginIvyDeps = Agg(ivy"org.chipsalliance:::chisel-plugin:6.7.0")
  override def scalacOptions       = Seq("-deprecation", "-feature", "-language:reflectiveCalls", "-Ymacro-annotations")
}

object queue       extends SbtModule {
  override def millSourcePath      = os.pwd / os.up / "queue"
  override def scalaVersion        = "2.13.16"
  override def ivyDeps             = Agg(ivy"org.chipsalliance::chisel:6.7.0")
  override def scalacPluginIvyDeps = Agg(ivy"org.chipsalliance:::chisel-plugin:6.7.0")
  override def scalacOptions       = Seq("-language:reflectiveCalls", "-Ymacro-annotations")
}

object chi       extends ChiselModule {
  override def moduleDeps = Seq(queue)
  override def moduleRoot = os.pwd / os.up / "chi"
}

object mmu       extends ChiselModule {
  override def moduleRoot = os.pwd / os.up / "mmu"
  override def moduleDeps = Seq(chi)
}

object preflight extends ChiselModule {
  override def moduleRoot = os.pwd
  override def moduleDeps = Seq(chi, mmu)
  override def mainClass  = Some("memcore.memory.preflight.Emit")
}
