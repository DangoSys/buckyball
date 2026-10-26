import mill._
import mill.scalalib._
import $file.^.^.^.arch.{build => design_build}
import $file.^.^.^.^.^.^.arch.src.main.scala.sims.{build => sims_build}

val entryRoot: os.Path = millSourcePath

object simulation extends SbtModule {
  override def millSourcePath = entryRoot
  override def scalaVersion = "2.13.16"
  override def moduleDeps = Seq(design_build.design, sims_build.library)
  override def ivyDeps = Agg(ivy"org.chipsalliance::chisel:6.7.0")
  override def scalacPluginIvyDeps = Agg(ivy"org.chipsalliance:::chisel-plugin:6.7.0")
  override def scalacOptions = Seq("-language:reflectiveCalls", "-Ymacro-annotations")
  override def sources = T.sources(entryRoot / "DeviceConfig.scala")
  override def mainClass = Some("sims.verilator.Elaborate")
}
