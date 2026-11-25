import mill._
import mill.scalalib._
import $file.^.^.^.^.^.devices.`multi-rocket`.arch.{build => design_build}

// The entry of the multi-rocket rocket hart alone.
val entryRoot: os.Path = millSourcePath

object simulation extends SbtModule {
  override def millSourcePath = entryRoot
  override def scalaVersion = "2.13.16"
  override def moduleDeps = Seq(design_build.design)
  override def ivyDeps = Agg(ivy"org.chipsalliance::chisel:6.7.0")
  override def scalacPluginIvyDeps = Agg(ivy"org.chipsalliance:::chisel-plugin:6.7.0")
  override def scalacOptions = Seq("-language:reflectiveCalls", "-Ymacro-annotations")
  override def sources = T.sources(entryRoot / "HartConfig.scala")
  override def mainClass = Some("framework.device.sims.verilator.Elaborate")
}
