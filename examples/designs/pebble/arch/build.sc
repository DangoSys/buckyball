import mill._
import mill.scalalib._
import $file.^.^.^.^.arch.src.main.scala.framework.device.cbus.{build => cbus_build}
import $file.^.^.^.{build => examples}

val designRoot: os.Path = millSourcePath / os.up

object pebble extends SbtModule {
  override def generatedSources    = T(cbus_build.registry(sources(), T.dest))
  override def millSourcePath      = designRoot
  override def forkWorkingDir      = T(designRoot)
  override def scalaVersion        = "2.13.16"
  override def moduleDeps          = Seq(examples.library)
  override def ivyDeps             = Agg(ivy"org.chipsalliance::chisel:6.7.0")
  override def scalacPluginIvyDeps = Agg(ivy"org.chipsalliance:::chisel-plugin:6.7.0")
  override def scalacOptions       = Seq("-deprecation", "-feature", "-language:reflectiveCalls", "-Ymacro-annotations")
  override def sources             = T.sources(designRoot / "arch")
}
