import mill._
import mill.scalalib._

trait ChiselModule extends SbtModule {
  def moduleRoot: os.Path

  override def millSourcePath = moduleRoot
  override def scalaVersion = "2.13.16"
  override def ivyDeps = Agg(ivy"org.chipsalliance::chisel:6.7.0")
  override def scalacPluginIvyDeps = Agg(ivy"org.chipsalliance:::chisel-plugin:6.7.0")
  override def scalacOptions = Seq("-deprecation", "-feature", "-language:reflectiveCalls", "-Ymacro-annotations")
}

val frameworkRoot = os.pwd / os.up / os.up
val memCoreRoot = frameworkRoot / "mem-core"

object axis extends ChiselModule {
  override def moduleRoot = memCoreRoot / "axis"
}

object chi extends ChiselModule {
  override def moduleRoot = memCoreRoot / "chi"
}

object bank extends ChiselModule {
  override def moduleRoot = memCoreRoot / "bank"
  override def moduleDeps = Seq(axis)
}

object cache extends ChiselModule {
  override def moduleRoot = memCoreRoot / "cache"
  override def moduleDeps = Seq(chi)
}

object coherence extends ChiselModule {
  override def moduleRoot = memCoreRoot / "coherence"
  override def moduleDeps = Seq(chi, cache)
}

object root_chip extends ChiselModule {
  override def moduleRoot = os.pwd
  override def moduleDeps = Seq(axis, chi, bank, coherence, cache)
}
