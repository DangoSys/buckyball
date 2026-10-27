import mill._
import mill.scalalib._
import $file.^.^.^.^.arch.src.main.scala.framework.device.cbus.{build => cbus_build}
import $file.^.^.^.{build => examples}
import $file.^.^.^.prefab.cores.rocket.arch.{build => rocket}
val designRoot: os.Path = millSourcePath / os.up

object design extends SbtModule {
  override def generatedSources    = T(cbus_build.registry(sources() ++ rocket.library.sources(), T.dest))
  override def forkWorkingDir      = T(designRoot / os.up / os.up / os.up / "arch")
  override def scalaVersion        = "2.13.16"
  override def moduleDeps          = Seq(examples.library, rocket.library)
  override def ivyDeps             = Agg(ivy"org.chipsalliance::chisel:6.7.0")
  override def scalacPluginIvyDeps = Agg(ivy"org.chipsalliance:::chisel-plugin:6.7.0")
  override def scalacOptions       = Seq("-language:reflectiveCalls", "-Ymacro-annotations")
  override def sources             = T.sources(designRoot / "arch", designRoot / "configs")
}
