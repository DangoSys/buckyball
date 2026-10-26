import mill._
import mill.scalalib._

val archRoot: os.Path = millSourcePath

trait FrameworkModule extends SbtModule {
  def moduleRoot: os.Path
  override def millSourcePath      = moduleRoot
  override def forkWorkingDir      = T(moduleRoot)
  override def scalaVersion        = "2.13.16"
  override def ivyDeps             = Agg(ivy"org.chipsalliance::chisel:6.7.0", ivy"com.lihaoyi::upickle:3.3.1")
  override def scalacPluginIvyDeps = Agg(ivy"org.chipsalliance:::chisel-plugin:6.7.0")
  override def scalacOptions       = Seq("-deprecation", "-feature", "-language:reflectiveCalls", "-Ymacro-annotations")
}

trait IpModule extends FrameworkModule {
  def framework: ScalaModule
  override def moduleDeps = Seq(framework)
  override def sources    = T.sources(Seq(PathRef(moduleRoot / "src" / "main" / "scala" / "Emit.scala")))
}

object lib extends FrameworkModule {
  override def moduleRoot = archRoot / "src" / "main" / "scala" / "framework" / "lib"

  override def sources    = T.sources {
    Seq(
      PathRef(moduleRoot / "queue"),
      PathRef(moduleRoot / "arbiter"),
      PathRef(moduleRoot / "arith"),
      PathRef(moduleRoot / "dpi")
    )
  }

}
