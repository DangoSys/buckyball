import mill._
import mill.scalalib._
import $file.^.framework.{build => framework_build}

val simRoot: os.Path = millSourcePath

object library extends SbtModule {
  override def millSourcePath = simRoot
  override def scalaVersion = "2.13.16"
  override def moduleDeps = Seq(framework_build.library)
  override def ivyDeps = Agg(ivy"org.chipsalliance::chisel:6.7.0", ivy"com.lihaoyi::upickle:3.3.1")
  override def scalacPluginIvyDeps = Agg(ivy"org.chipsalliance:::chisel-plugin:6.7.0")
  override def scalacOptions = Seq("-language:reflectiveCalls", "-deprecation", "-feature", "-Ymacro-annotations")
  override def sources = T.sources {
    os.walk(simRoot).filter(path => path.ext == "scala" && !path.toString.contains("/out/")).map(PathRef(_))
  }
}
