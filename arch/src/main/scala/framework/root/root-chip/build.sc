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

val frameworkRoot = os.pwd / os.up / os.up
val memCoreRoot   = frameworkRoot / "mem-core"

object queue extends SbtModule {
  override def millSourcePath      = os.pwd / os.up / os.up / "mem-core" / "queue"
  override def scalaVersion        = "2.13.16"
  override def ivyDeps             = Agg(ivy"org.chipsalliance::chisel:6.7.0")
  override def scalacPluginIvyDeps = Agg(ivy"org.chipsalliance:::chisel-plugin:6.7.0")
  override def scalacOptions       = Seq("-language:reflectiveCalls", "-Ymacro-annotations")
}

object axis extends ChiselModule {
  override def moduleRoot = memCoreRoot / "axis"
}

object chi extends ChiselModule {
  override def moduleDeps = Seq(queue)
  override def moduleRoot = memCoreRoot / "chi"
}

object bank extends ChiselModule {
  override def moduleRoot = memCoreRoot / "bank"
  override def moduleDeps = Seq(axis)
}

object root_chip extends ChiselModule {
  override def moduleRoot = os.pwd
  override def mainClass  = Some("hier.chip.mesh.Emit")
  override def moduleDeps = Seq(axis, chi, bank)
}
