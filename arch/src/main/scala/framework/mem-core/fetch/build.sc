import mill._
import mill.scalalib._
object fetch extends SbtModule {
  override def millSourcePath = os.pwd
  override def scalaVersion = "2.13.16"
  override def ivyDeps = Agg(ivy"org.chipsalliance::chisel:6.7.0")
  override def scalacPluginIvyDeps = Agg(ivy"org.chipsalliance:::chisel-plugin:6.7.0")
  override def scalacOptions = Seq("-deprecation", "-feature", "-language:reflectiveCalls", "-Ymacro-annotations")
  override def mainClass = Some("memcore.memory.fetch.Emit")
}
