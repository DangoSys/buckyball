import mill._
import mill.scalalib._
import $file.^.^.^.^.^.{build => examples}

// The entry of the prefab attention thread alone: every prefab thread is on its classpath.
val entryRoot: os.Path = millSourcePath

object simulation extends SbtModule {
  override def millSourcePath = entryRoot
  override def scalaVersion = "2.13.16"
  override def moduleDeps = Seq(examples.library)
  override def ivyDeps = Agg(ivy"org.chipsalliance::chisel:6.7.0")
  override def scalacPluginIvyDeps = Agg(ivy"org.chipsalliance:::chisel-plugin:6.7.0")
  override def scalacOptions = Seq("-language:reflectiveCalls", "-Ymacro-annotations")
  override def sources = T.sources(Seq.empty[PathRef])
  override def mainClass = Some("framework.thread.sims.Elaborate")
}
