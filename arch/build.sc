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
  override def scalacOptions = Seq("-deprecation", "-feature", "-language:reflectiveCalls", "-Ymacro-annotations")
}

val frameworkRoot = os.pwd / "src" / "main" / "scala" / "framework"

object queue extends FrameworkModule {
  override def moduleRoot = frameworkRoot / "mem-core" / "queue"
}

object axis extends FrameworkModule {
  override def moduleRoot = frameworkRoot / "mem-core" / "axis"
  override def scalacOptions = super.scalacOptions() ++ Seq("-Ymacro-annotations")
}

object axi4 extends FrameworkModule {
  override def moduleDeps = Seq(queue)
  override def moduleRoot = frameworkRoot / "mem-core" / "axi4"
  override def mainClass = Some("memcore.bus.axi4.Emit")
}

object ddr extends FrameworkModule {
  override def moduleRoot = frameworkRoot / "mem-core" / "ddr"
  override def moduleDeps = Seq(chi, axi4)
  override def mainClass = Some("memcore.memory.ddr.Emit")
}

object chi extends FrameworkModule {
  override def moduleDeps = Seq(queue)
  override def moduleRoot = frameworkRoot / "mem-core" / "chi"
  override def scalacOptions = super.scalacOptions() ++ Seq("-Ymacro-annotations")
}

object cpu_mem extends FrameworkModule {
  override def moduleRoot = frameworkRoot / "mem-core" / "cpu_mem"
  override def moduleDeps = Seq(chi, mmu)
}

object mmu extends FrameworkModule {
  override def moduleRoot = frameworkRoot / "mem-core" / "mmu"
  override def moduleDeps = Seq(chi)
}

object fetch extends FrameworkModule {
  override def moduleRoot = frameworkRoot / "mem-core" / "fetch"
}

object preflight extends FrameworkModule {
  override def moduleRoot = frameworkRoot / "mem-core" / "preflight"
  override def moduleDeps = Seq(chi, mmu)
  override def mainClass = Some("memcore.memory.preflight.Emit")
}

object interlock extends FrameworkModule {
  override def moduleRoot = frameworkRoot / "mem-core" / "interlock"
  override def mainClass = Some("memcore.memory.interlock.Emit")
}

object bank extends FrameworkModule {
  override def moduleRoot = frameworkRoot / "mem-core" / "bank"
  override def moduleDeps = Seq(axis)
  override def scalacOptions = super.scalacOptions() ++ Seq("-Ymacro-annotations")
}

object cache extends FrameworkModule {
  override def moduleDeps = Seq(queue)
  override def moduleRoot = frameworkRoot / "mem-core" / "cache"
  override def sources = T.sources { super.sources() ++ Seq(PathRef(moduleRoot / "configs")) }
  override def ivyDeps = super.ivyDeps() ++ Agg(ivy"tech.sparse::toml-scala:0.2.2")
  override def scalacOptions = super.scalacOptions() ++ Seq("-Ymacro-annotations")
}

object mesh_shm extends FrameworkModule {
  override def scalacOptions = super.scalacOptions() ++ Seq("-Ymacro-annotations")
  override def moduleRoot = frameworkRoot / "mem-core" / "mesh_shm"
  override def moduleDeps = Seq(bank)
}

object coherence extends FrameworkModule {
  override def moduleRoot = frameworkRoot / "mem-core" / "coherence"
  override def moduleDeps = Seq(chi, cache)
  override def sources = T.sources { super.sources() ++ Seq(PathRef(moduleRoot / "configs")) }
  override def scalacOptions = super.scalacOptions() ++ Seq("-Ymacro-annotations")
}

object blink extends FrameworkModule {
  override def moduleRoot = frameworkRoot / "balldomain" / "blink"
  override def moduleDeps = Seq(rocket_bb, ant, coherence, cpu_mem, ddr, interlock)
  override def ivyDeps = super.ivyDeps() ++ Agg(ivy"com.lihaoyi::upickle:3.3.1")
  override def scalacOptions = super.scalacOptions() ++ Seq("-Ymacro-annotations")
  override def sources = T.sources {
    Seq(
      PathRef(frameworkRoot / "top" / "GlobalConfig.scala"),
      PathRef(frameworkRoot / "top" / "configs" / "DesignConfig.scala"),
      PathRef(frameworkRoot / "top" / "configs" / "SimParam.scala"),
      PathRef(frameworkRoot / "memdomain" / "configs" / "MemDomainParam.scala"),
      PathRef(frameworkRoot / "frontend" / "configs" / "FrontendParam.scala"),
      PathRef(frameworkRoot / "rvv" / "src" / "main" / "scala" / "configs" / "RvvParam.scala"),
      PathRef(frameworkRoot / "balldomain" / "configs" / "BallDomainParam.scala"),
      PathRef(frameworkRoot / "system" / "tile" / "configs" / "TileParam.scala"),
      PathRef(frameworkRoot / "balldomain" / "blink" / "blink.scala"),
      PathRef(frameworkRoot / "balldomain" / "blink" / "bank.scala"),
      PathRef(frameworkRoot / "balldomain" / "blink" / "status.scala"),
      PathRef(frameworkRoot / "balldomain" / "blink" / "baseball.scala"),
      PathRef(frameworkRoot / "balldomain" / "blink" / "SubRobRow.scala"),
      PathRef(frameworkRoot / "balldomain" / "blink" / "mmio" / "MmioRead.scala"),
      PathRef(frameworkRoot / "balldomain" / "blink" / "mmio" / "MmioWrite.scala"),
      PathRef(frameworkRoot / "memdomain" / "backend" / "banks" / "SramIO.scala"),
      PathRef(frameworkRoot / "memdomain" / "backend" / "mmio" / "MmioIO.scala"),
      PathRef(frameworkRoot / "balldomain" / "rs" / "interfaces.scala"),
      PathRef(frameworkRoot / "balldomain" / "decoder" / "BallDecodeCmd.scala"),
      PathRef(frameworkRoot / "frontend" / "decoder" / "PostGDCmd.scala"),
      PathRef(frameworkRoot / "frontend" / "scoreboard" / "BankAccessInfo.scala")
    )
  }
}

object arith extends FrameworkModule {
  override def moduleRoot = frameworkRoot / "arith"
}

object spm extends FrameworkModule {
  override def moduleRoot = frameworkRoot / "mem-core" / "spm"
  override def moduleDeps = Seq(bank)
}

object tss extends FrameworkModule {
  override def moduleRoot = frameworkRoot / "mem-core" / "tss"
  override def moduleDeps = Seq(spm)
  override def mainClass = Some("memcore.memory.tss.Emit")
}

object tls extends FrameworkModule {
  override def moduleRoot = frameworkRoot / "mem-core" / "tls"
  override def moduleDeps = Seq(spm)
  override def mainClass = Some("memcore.memory.tls.Emit")
}

object ant extends FrameworkModule {
  override def moduleRoot = frameworkRoot / "system" / "core" / "ant"
  override def moduleDeps = Seq(tls, tss, arith)
  override def mainClass = Some("framework.ant.Emit")
}

object rvv extends FrameworkModule {
  override def moduleRoot = frameworkRoot / "rvv"
  override def moduleDeps = Seq(hardfloat, blink, arith)
  override def sources = T.sources {
    os.walk(moduleRoot / "src" / "main" / "scala")
      .filter(path => path.ext == "scala" && path.last != "RvvParam.scala")
      .map(PathRef(_))
  }
  override def scalacOptions = super.scalacOptions() ++ Seq("-Ymacro-annotations")
}

object seed extends FrameworkModule {
  override def moduleRoot = frameworkRoot / "system" / "core" / "seed"
}

object rocket_bb extends FrameworkModule {
  override def moduleRoot = frameworkRoot / "system" / "core" / "rocket"
  override def moduleDeps = Seq(rocketchip)
  override def ivyDeps = super.ivyDeps() ++ Agg(ivy"com.lihaoyi::upickle:3.3.1")
  override def sources = T.sources {
    os.walk(moduleRoot)
      .filter(path => path.ext == "scala" && !Set("CoreParameters.scala", "CpuInterfaces.scala", "TileInterrupts.scala", "TileParameters.scala", "RocketCoreParameters.scala", "Utility.scala", "VectorInterfaces.scala", "RocketCpuParam.scala").contains(path.last))
      .map(PathRef(_))
  }
}

object root_chip extends FrameworkModule {
  override def moduleRoot = frameworkRoot / "root" / "root-chip"
  override def moduleDeps = Seq(axis, chi, bank)
}

object root_core extends FrameworkModule {
  override def mainClass = Some("hier.core.rocket.Emit")
  override def moduleRoot = frameworkRoot / "root" / "root-core"
  override def moduleDeps = Seq(rocket_bb, fetch, cpu_mem, chi, coherence, interlock, preflight)
  override def sources = T.sources {
    super.sources() ++ Seq(
      PathRef(os.pwd / os.up / "examples" / "cores" / "rocket" / "arch" / "src" / "main" / "scala" / "cpu"),
      PathRef(frameworkRoot / "system" / "core" / "clink" / "base.scala")
    )
  }
}

object root_tile extends FrameworkModule {
  override def mainClass = Some("hier.tile.memory.Emit")
  override def moduleRoot = frameworkRoot / "root" / "root-tile"
  override def moduleDeps = Seq(chi, coherence, cpu_mem, mmu, root_core, ddr)
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

  override def moduleDeps = Seq(
    rocketchip,
    gemmini,
    protoJava,
    axis,
    fetch,
    chi,
    bank,
    mesh_shm,
    cache,
    coherence,
    ddr,
    rvv,
    seed,
    ant,
    rocket_bb,
    root_chip,
    root_core,
    root_tile
  )

  override def sources = T.sources {
    val examples = os.pwd / os.up / "examples"
    def archSrcs(kind: String) =
      os.list(examples / kind)
        .filter(os.isDir)
        .map(_ / "arch" / "src" / "main" / "scala")
        .filter(os.exists)
        .flatMap(root => os.walk(root).filter(path => path.ext == "scala"))
        .map(PathRef(_))

    def configSrcs(kind: String) =
      os.list(examples / kind)
        .filter(os.isDir)
        .map(_ / "configs")
        .filter(os.exists)
        .map(PathRef(_))

    val sharedBlink = blink.sources().map(_.path).toSet
    val localSources = os.walk(os.pwd / "src" / "main" / "scala")
      .filter(path => path.ext == "scala")
      .filterNot(sharedBlink.contains)
      .filterNot(path => path.toString.contains("/framework/root/"))
      .filterNot(path => path.toString.contains("/verification/"))
      .filterNot(path => path.toString.contains("/framework/mem-core/"))
      .filterNot(path => path.toString.contains("/framework/rvv/"))
      .filterNot(path => path.toString.contains("/framework/system/core/seed/"))
      .filterNot(path => path.toString.contains("/framework/system/core/ant/"))
      .filterNot(path => path.toString.contains("/framework/arith/"))
      .filterNot(path => path.toString.contains("/framework/system/core/rocket/"))
      .filterNot(path => path == frameworkRoot / "system" / "core" / "clink" / "base.scala")
      .map(PathRef(_))
    val coreSources = archSrcs("cores")
      .filterNot(_.path.toString.contains("/cores/rocket/arch/src/main/scala/cpu/"))
    localSources ++ archSrcs("balls") ++ coreSources ++ archSrcs("chips") ++
      configSrcs("balls") ++ configSrcs("chips") ++ configSrcs("cores")
  }

  override def ivyDeps = Agg(
    ivy"org.chipsalliance::chisel:6.7.0",
    ivy"org.apache.commons:commons-lang3:3.12.0",
    ivy"org.apache.commons:commons-text:1.9",
    ivy"org.yaml:snakeyaml:2.0",
    ivy"com.lihaoyi::sourcecode:0.3.0",
    ivy"com.lihaoyi::upickle:3.3.1",
    ivy"tech.sparse::toml-scala:0.2.2",
    ivy"com.google.protobuf:protobuf-java:4.35.1"
  )

  override def scalacPluginIvyDeps = Agg(
    ivy"org.chipsalliance:::chisel-plugin:6.7.0"
  )

  object test extends ScalaModule with TestModule.ScalaTest {
    override def scalaVersion = T("2.13.16")
    override def moduleDeps = Seq(m)

    override def ivyDeps = Agg(
      ivy"org.scalatest::scalatest::3.2.19"
    )

  }

}

// Define cde module - must be compiled first
object cde extends SbtModule {
  override def millSourcePath =
    os.pwd / "thirdparty" / "rocket-chip" / "dependencies" / "cde"
  override def scalaVersion = "2.13.16"

  override def sources = T.sources {
    Seq(PathRef(millSourcePath / "cde" / "src"))
  }

  override def ivyDeps = Agg(
    ivy"org.chipsalliance::chisel:6.7.0"
  )

  override def scalacPluginIvyDeps = Agg(
    ivy"org.chipsalliance:::chisel-plugin:6.7.0"
  )

}

// Define hardfloat module
object hardfloat extends SbtModule {
  override def millSourcePath =
    os.pwd / "thirdparty" / "berkeley-hardfloat"
  override def scalaVersion = "2.13.16"

  // Override sources to match build.sbt behavior
  override def sources = T.sources {
    super.sources() ++ Seq(
      PathRef(millSourcePath / "hardfloat" / "src" / "main" / "scala")
    )
  }

  override def ivyDeps = Agg(
    ivy"org.chipsalliance::chisel:6.7.0"
  )

  override def scalacPluginIvyDeps = Agg(
    ivy"org.chipsalliance:::chisel-plugin:6.7.0"
  )

}

// Define midas_target_utils module
object midas_target_utils extends SbtModule {
  override def millSourcePath =
    os.pwd / os.up / "thirdparty" / "firesim" / "sim" / "midas" / "targetutils"
  override def scalaVersion = "2.13.16"

  override def ivyDeps = Agg(
    ivy"org.chipsalliance::chisel:6.7.0"
  )

  override def scalacPluginIvyDeps = Agg(
    ivy"org.chipsalliance:::chisel-plugin:6.7.0"
  )

}

// Define rocket-chip module with proper dependencies
object rocketchip extends SbtModule {
  override def millSourcePath =
    os.pwd / "thirdparty" / "rocket-chip"
  override def scalaVersion = "2.13.16"

  override def sources = T.sources {
    val src = millSourcePath / "src" / "main" / "scala"
    val files = Seq(
      "rocket/ALU.scala", "rocket/AMOALU.scala", "rocket/BTB.scala", "rocket/Breakpoint.scala",
      "rocket/CSR.scala", "rocket/Consts.scala", "rocket/CustomInstructions.scala", "rocket/DebugROB.scala",
      "rocket/Decode.scala", "rocket/Events.scala", "rocket/IBuf.scala", "rocket/IDecode.scala",
      "rocket/Instructions.scala", "rocket/Instructions32.scala", "rocket/Multiplier.scala", "rocket/PMP.scala",
      "rocket/RVC.scala", "rocket/package.scala", "tile/FPU.scala", "tile/CustomCSRs.scala",
      "util/Misc.scala", "util/ClockGate.scala", "util/CoreMonitor.scala", "util/Property.scala",
      "util/Counters.scala", "util/ECC.scala", "util/Replacement.scala", "util/ShiftReg.scala",
      "util/ShiftQueue.scala", "util/HellaQueue.scala", "util/LCG.scala", "util/PlusArg.scala",
      "util/MuxLiteral.scala", "util/Arbiters.scala", "util/AsyncResetReg.scala", "util/Timer.scala",
      "util/GeneratorUtils.scala", "unittest/UnitTest.scala"
    )
    val adapter = frameworkRoot / "system" / "core" / "rocket"
    files.map(f => PathRef(src / os.RelPath(f))) ++ Seq(
      PathRef(adapter / "CoreParameters.scala"), PathRef(adapter / "CpuInterfaces.scala"),
      PathRef(adapter / "TileInterrupts.scala"), PathRef(adapter / "TileParameters.scala"),
      PathRef(adapter / "RocketCoreParameters.scala"), PathRef(adapter / "Utility.scala"),
      PathRef(adapter / "VectorInterfaces.scala"), PathRef(adapter / "configs" / "RocketCpuParam.scala")
    )
  }

  override def moduleDeps = Seq(cde, hardfloat, midas_target_utils)

  override def ivyDeps = Agg(
    ivy"org.chipsalliance::chisel:6.7.0",
    ivy"com.lihaoyi::upickle:3.3.1",
    ivy"com.lihaoyi::mainargs:0.5.0",
    ivy"org.json4s::json4s-jackson:4.0.5",
    ivy"org.scala-graph::graph-core:1.13.5"
  )

  override def scalacPluginIvyDeps = Agg(
    ivy"org.chipsalliance:::chisel-plugin:6.7.0"
  )

}


// Define gemmini module
object gemmini extends SbtModule {
  override def millSourcePath =
    os.pwd / "thirdparty" / "gemmini"
  override def scalaVersion = "2.13.16"

  override def sources = T.sources {
    Seq("Arithmetic", "Dataflow", "MeshWithDelays", "Mesh", "Tile", "PE", "Util", "Transposer", "TagQueue", "Pipeline", "Shifter", "SyncMem")
      .map(name => PathRef(millSourcePath / "src" / "main" / "scala" / "gemmini" / s"$name.scala"))
  }
  override def moduleDeps = Seq(hardfloat)

  override def ivyDeps = Agg(
    ivy"org.chipsalliance::chisel:6.7.0"
  )

  override def scalacPluginIvyDeps = Agg(
    ivy"org.chipsalliance:::chisel-plugin:6.7.0"
  )

}


object memdomain_ack extends FrameworkModule {
  override def moduleRoot = frameworkRoot / "memdomain" / "verification"
  override def moduleDeps = Seq(buckyball, ddr, preflight)
  override def mainClass = Some("framework.memdomain.verification.Emit")
}
