import mill._
import mill.scalalib._
import $file.^.^.^.^.^.^.^.arch.src.main.scala.hart.cbus.{build => cbus_build}
import $file.^.^.^.^.^.^.{build => examples}
import $file.^.^.^.arch.{build => reduce}

// The entry of the reduce hart alone: the hart and its registry.
val entryRoot: os.Path = millSourcePath

object simulation extends SbtModule {
  override def millSourcePath      = entryRoot
  override def generatedSources    = T(cbus_build.registry(reduce.library.sources(), T.dest))
  override def scalaVersion        = "2.13.16"
  override def moduleDeps          = Seq(examples.library, reduce.library)
  override def ivyDeps             = Agg(ivy"org.chipsalliance::chisel:6.7.0")
  override def scalacPluginIvyDeps = Agg(ivy"org.chipsalliance:::chisel-plugin:6.7.0")
  override def scalacOptions       = Seq("-language:reflectiveCalls", "-Ymacro-annotations")
  override def sources             = T.sources(entryRoot / "HartConfig.scala")
  override def mainClass           = Some("framework.device.sims.verilator.Elaborate")
}
