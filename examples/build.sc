import mill._
import mill.scalalib._
import $file.^.arch.src.main.scala.framework.{build => framework_build}
import $file.^.arch.thirdparty.{build => thirdparty_build}
import $file.components.balls.{build => ball_components}
import $file.components.threads.{build => thread_components}

import $file.^.arch.src.main.scala.framework.acc.balldomain.src.main.scala.bbus.{build => bbus_build}
import $file.^.arch.src.main.scala.framework.core.tbus.{build => tbus_build}

val examplesRoot: os.Path = millSourcePath

object library extends SbtModule {
  override def scalaVersion = "2.13.16"

  override def moduleDeps = Seq(
    framework_build.library,
    thirdparty_build.hardfloat,
    thirdparty_build.midas_target_utils,
    thirdparty_build.gemmini
  )

  override def ivyDeps = Agg(
    ivy"org.chipsalliance::chisel:6.7.0",
    ivy"com.lihaoyi::upickle:3.3.1",
    ivy"com.lihaoyi::mainargs:0.5.0",
    ivy"org.json4s::json4s-jackson:4.0.5",
    ivy"org.scala-graph::graph-core:1.13.5"
  )

  override def scalacPluginIvyDeps = Agg(ivy"org.chipsalliance:::chisel-plugin:6.7.0")
  override def scalacOptions       = Seq("-deprecation", "-feature", "-language:reflectiveCalls", "-Ymacro-annotations")

  override def generatedSources = T {
    bbus_build.registry(ball_components.sources(), examplesRoot / "components" / "balls", T.dest) ++
      tbus_build.registry(thread_components.sources(), examplesRoot / "components" / "threads", T.dest)
  }

  override def sources =
    T.sources(ball_components.sources() ++ thread_components.sources())
}
