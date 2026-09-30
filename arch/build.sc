import mill._
import mill.define.Sources
import mill.modules.Util
import mill.scalalib.TestModule.ScalaTest
import scalalib._
import mill.javalib.JavaModule
import mill.bsp._

object protoJava extends JavaModule {
  def protoDir = T {
    os.pwd / os.up / "bbdev" / "api" / "steps" / "config" / "scripts" / "proto"
  }

  def protoSources = T.sources {
    Seq(PathRef(protoDir() / "chip.proto"))
  }

  def generatedSources = T {
    val dir = protoDir()
    val proto = protoSources().head.path
    os.proc("protoc", s"-I$dir", s"--java_out=${T.dest}", proto).call()
    Seq(PathRef(T.dest))
  }

  override def zincIncrementalCompilation = T { false }

  override def ivyDeps = Agg(ivy"com.google.protobuf:protobuf-java:4.35.1")
}

trait FrameworkModule extends SbtModule {
  def moduleRoot: os.Path

  override def millSourcePath = moduleRoot
  override def scalaVersion = "2.13.16"
  override def ivyDeps = Agg(ivy"org.chipsalliance::chisel:6.7.0")
  override def scalacPluginIvyDeps = Agg(ivy"org.chipsalliance:::chisel-plugin:6.7.0")
  override def scalacOptions = Seq("-deprecation", "-feature", "-language:reflectiveCalls")
}

val frameworkRoot = os.pwd / "src" / "main" / "scala" / "framework"

object axis extends FrameworkModule {
  override def moduleRoot = frameworkRoot / "mem-core" / "axis"
  override def scalacOptions = super.scalacOptions() ++ Seq("-Ymacro-annotations")
}

object chi extends FrameworkModule {
  override def moduleRoot = frameworkRoot / "mem-core" / "chi"
  override def scalacOptions = super.scalacOptions() ++ Seq("-Ymacro-annotations")
}

object bank extends FrameworkModule {
  override def moduleRoot = frameworkRoot / "mem-core" / "bank"
  override def moduleDeps = Seq(axis)
  override def scalacOptions = super.scalacOptions() ++ Seq("-Ymacro-annotations")
}

object cache extends FrameworkModule {
  override def moduleRoot = frameworkRoot / "mem-core" / "cache"
  override def sources = T.sources { super.sources() ++ Seq(PathRef(moduleRoot / "configs")) }
  override def ivyDeps = super.ivyDeps() ++ Agg(ivy"tech.sparse::toml-scala:0.2.2")
  override def scalacOptions = super.scalacOptions() ++ Seq("-Ymacro-annotations")
}

object mesh_shm extends FrameworkModule {
  override def moduleRoot = frameworkRoot / "mem-core" / "mesh_shm"
  override def moduleDeps = Seq(bank)
}

object coherence extends FrameworkModule {
  override def moduleRoot = frameworkRoot / "mem-core" / "coherence"
  override def moduleDeps = Seq(chi, cache)
  override def sources = T.sources { super.sources() ++ Seq(PathRef(moduleRoot / "configs")) }
  override def scalacOptions = super.scalacOptions() ++ Seq("-Ymacro-annotations")
}

object rvv extends FrameworkModule {
  override def moduleRoot = frameworkRoot / "rvv"
  override def moduleDeps = Seq(hardfloat)
  override def scalacOptions = super.scalacOptions() ++ Seq("-Ymacro-annotations")
}

object seed extends FrameworkModule {
  override def moduleRoot = frameworkRoot / "system" / "core" / "seed"
}

object root_chip extends FrameworkModule {
  override def moduleRoot = frameworkRoot / "root" / "root-chip"
  override def moduleDeps = Seq(axis, chi, bank, coherence)
}

object root_core extends FrameworkModule {
  override def moduleRoot = frameworkRoot / "root" / "root-core"
  override def moduleDeps = Seq(axis)
}

object root_tile extends FrameworkModule {
  override def moduleRoot = frameworkRoot / "root" / "root-tile"
  override def moduleDeps = Seq(axis)
}

object buckyball extends SbtModule { m =>
  override def millSourcePath = os.pwd
  override def scalaVersion = "2.13.16"

  override def scalacOptions = Seq(
    "-language:reflectiveCalls",
    "-deprecation",
    "-feature",
    "-Xcheckinit",
    "-Ymacro-annotations"
  )

  // Add chipyard and rocket-chip dependencies
  override def moduleDeps = Seq(
    chipyard,
    gemmini,
    protoJava,
    axis,
    chi,
    bank,
    mesh_shm,
