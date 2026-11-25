import mill._
import mill.scalalib._
import $file.^.^.^.^.{build => examples}

val componentRoot: os.Path = millSourcePath / os.up

object library extends SbtModule {
  override def scalaVersion        = "2.13.16"
  override def moduleDeps          = Seq(examples.library)
  override def ivyDeps             = Agg(ivy"org.chipsalliance::chisel:6.7.0")
  override def scalacPluginIvyDeps = Agg(ivy"org.chipsalliance:::chisel-plugin:6.7.0")
  override def scalacOptions       = Seq("-language:reflectiveCalls", "-Ymacro-annotations")
  override def sources             = T.sources(componentRoot / "arch")
}
