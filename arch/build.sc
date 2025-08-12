// import Mill dependency
import mill._
import mill.define.Sources
import mill.modules.Util
import mill.scalalib.TestModule.ScalaTest
import scalalib._
// support BSP
import mill.bsp._



object main extends SbtModule { m =>
  override def millSourcePath = os.pwd
  override def scalaVersion = "2.13.12"
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
    chipyardGenerators
  )
  
  override def ivyDeps = Agg(
    // ivy"org.chipsalliance::chisel:6.5.0",
    ivy"org.chipsalliance::chisel:6.5.0",
    ivy"org.apache.commons:commons-lang3:3.12.0",
    ivy"org.apache.commons:commons-text:1.9",
    // ivy"org.chipsalliance::circt:1.0.0",
    // ivy"org.chipsalliance::circt-mlir:1.0.0"
    ivy"org.yaml:snakeyaml:2.0",
    ivy"com.lihaoyi::sourcecode:0.3.0"
  )
  
  override def scalacPluginIvyDeps = Agg(
    ivy"org.chipsalliance:::chisel-plugin:6.5.0"
    // ivy"org.chipsalliance:::chisel-plugin:7.0.0-RC1",
    // ivy"org.chipsalliance:::chisel-plugin:6.5.0"
  )
  
  object test extends ScalaModule with TestModule.ScalaTest {
    override def scalaVersion = T("2.13.12")
    override def moduleDeps = Seq(m)
    override def ivyDeps = Agg(
      ivy"org.scalatest::scalatest::3.2.19"
      // ivy"org.scalatest::scalatest:3.2.16"
    )
  }
}

// Define cde module - must be compiled first
object cde extends SbtModule {
  override def millSourcePath = os.pwd / os.up / "thirdparty" / "chipyard" / "tools" / "cde"
  override def scalaVersion = "2.13.12"
  
  // Override sources to match freshProject behavior
  override def sources = T.sources {
    super.sources() ++ Seq(PathRef(millSourcePath / "cde" / "src" / "chipsalliance"))
  }
  
  override def ivyDeps = Agg(
    ivy"org.chipsalliance::chisel:6.5.0"
  )
  
  override def scalacPluginIvyDeps = Agg(
    ivy"org.chipsalliance:::chisel-plugin:6.5.0"
  )
}

// Define hardfloat module - depends on cde
object hardfloat extends SbtModule {
  override def millSourcePath = os.pwd / os.up / "thirdparty" / "chipyard" / "generators" / "hardfloat"
  override def scalaVersion = "2.13.12"
  
  // Add cde dependency
  override def moduleDeps = Seq(
    cde
  )
  
  // Override sources to match build.sbt behavior
  override def sources = T.sources {
    super.sources() ++ Seq(PathRef(millSourcePath / "hardfloat" / "src" / "main" / "scala"))
  }
  
  override def ivyDeps = Agg(
    ivy"org.chipsalliance::chisel:6.5.0"
  )
  
  override def scalacPluginIvyDeps = Agg(
    ivy"org.chipsalliance:::chisel-plugin:6.5.0"
  )
}

// Define midas_target_utils module
object midasTargetUtils extends SbtModule {
  override def millSourcePath = os.pwd / os.up / "thirdparty" / "chipyard" / "sims" / "firesim" / "sim" / "midas" / "targetutils"
  override def scalaVersion = "2.13.12"
  
  override def ivyDeps = Agg(
    ivy"org.chipsalliance::chisel:6.5.0"
  )
  
  override def scalacPluginIvyDeps = Agg(
    ivy"org.chipsalliance:::chisel-plugin:6.5.0"
  )
}

// Define diplomacy module - depends on cde
object diplomacy extends SbtModule {
  override def millSourcePath = os.pwd / os.up / "thirdparty" / "chipyard" / "generators" / "diplomacy" / "diplomacy"
  override def scalaVersion = "2.13.12"
  
  // Add cde dependency first
  override def moduleDeps = Seq(
    cde
  )
  
  // Override sources to match freshProject behavior
  override def sources = T.sources {
    super.sources() ++ Seq(PathRef(millSourcePath / "src" / "diplomacy"))
  }
  
  override def ivyDeps = Agg(
    ivy"org.chipsalliance::chisel:6.5.0",
    ivy"com.lihaoyi::sourcecode:0.3.0"
  )
  
  override def scalacPluginIvyDeps = Agg(
    ivy"org.chipsalliance:::chisel-plugin:6.5.0"
  )
}

// Define rocket-chip module with proper dependencies
object rocketChip extends SbtModule {
  override def millSourcePath = os.pwd / os.up / "thirdparty" / "chipyard" / "generators" / "rocket-chip"
  override def scalaVersion = "2.13.12"
  
  // Add required dependencies for rocket-chip
  override def moduleDeps = Seq(
    diplomacy,
    cde,
    hardfloat,
    midasTargetUtils
  )
  
  override def ivyDeps = Agg(
    ivy"org.chipsalliance::chisel:6.5.0",
    ivy"com.lihaoyi::mainargs:0.5.0",
    ivy"org.json4s::json4s-jackson:4.0.5",
    ivy"org.scala-graph::graph-core:1.13.5"
  )
  
  override def scalacPluginIvyDeps = Agg(
    ivy"org.chipsalliance:::chisel-plugin:6.5.0"
  )
}

// Define chipyard module
object chipyard extends SbtModule {
  override def millSourcePath = os.pwd / os.up / "thirdparty" / "chipyard"
  override def scalaVersion = "2.13.12"
  
  // Override sources to include tools/stage, generators/chipyard, and harness directories (as per build.sbt)
  override def sources = T.sources {
    super.sources() ++ Seq(
      PathRef(millSourcePath / "tools" / "stage" / "src" / "main" / "scala"),
      PathRef(millSourcePath / "generators" / "chipyard" / "src" / "main" / "scala"),
      PathRef(millSourcePath / "generators" / "chipyard" / "src" / "main" / "scala" / "harness")
    )
  }
  
  // Add all required dependencies as per build.sbt
  override def moduleDeps = Seq(
    testchipip,
    rocketChip,
    boom,
    rocketChipBlocks,
    rocketchipInclusiveCache,
    dsptools,
    rocketDspUtils,
    gemmini,
    icenet,
    tracegen,
    cva6,
    nvdla,
    sodor,
    ibex,
    fftgenerator,
    constellation,
    mempress,
    barf,
    shuttle,
    caliptraAes,
    rerocc,
    compressacc,
    saturn,
    ara,
    firrtl2Bridge,
    vexiiriscv
  )
  
  override def ivyDeps = Agg(
    ivy"org.chipsalliance::chisel:6.5.0",
    ivy"org.reflections:reflections:0.10.2"
  )
  
  override def scalacPluginIvyDeps = Agg(
    ivy"org.chipsalliance:::chisel-plugin:6.5.0"
  )
}

// Define chipyard generators module
object chipyardGenerators extends SbtModule {
  override def millSourcePath = os.pwd / os.up / "thirdparty" / "chipyard" / "generators" / "chipyard"
  override def scalaVersion = "2.13.12"
  
  // Add chipyard, testchipip, and other required modules as dependencies
  override def moduleDeps = Seq(
    chipyard,
    testchipip,
    icenet,
    nvdla,
    fftgenerator,
    constellation,
    rerocc,
    rocketDspUtils,
    dsptools,
    fixedpoint,
    tracegen,
    boom,
    shuttle,
    ara,
    saturn,
    gemmini,
    sodor,
    compressacc,
    mempress,
    barf,
    caliptraAes,
    roccAccUtils,
    vexiiriscv,
    ibex,
    cva6
  )
  
  override def ivyDeps = Agg(
    ivy"org.chipsalliance::chisel:6.5.0",
    ivy"org.reflections:reflections:0.10.2"
  )
  
  override def scalacPluginIvyDeps = Agg(
    ivy"org.chipsalliance:::chisel-plugin:6.5.0"
  )
}

// Define testchipip module
object testchipip extends SbtModule {
  override def millSourcePath = os.pwd / os.up / "thirdparty" / "chipyard" / "generators" / "testchipip"
  override def scalaVersion = "2.13.12"
  
  // Add rocket-chip and rocket-chip-blocks as dependencies
  override def moduleDeps = Seq(
    rocketChip,
    rocketChipBlocks
  )
  
  override def ivyDeps = Agg(
    ivy"org.chipsalliance::chisel:6.5.0"
  )
  
  override def scalacPluginIvyDeps = Agg(
    ivy"org.chipsalliance:::chisel-plugin:6.5.0"
  )
}

// Define rocket-chip-blocks module (contains sifive package)
object rocketChipBlocks extends SbtModule {
  override def millSourcePath = os.pwd / os.up / "thirdparty" / "chipyard" / "generators" / "rocket-chip-blocks"
  override def scalaVersion = "2.13.12"
  
  // Add rocket-chip as a dependency
  override def moduleDeps = Seq(
    rocketChip
  )
  
  override def ivyDeps = Agg(
    ivy"org.chipsalliance::chisel:6.5.0"
  )
  
  override def scalacPluginIvyDeps = Agg(
    ivy"org.chipsalliance:::chisel-plugin:6.5.0"
  )
}

// Define icenet module
object icenet extends SbtModule {
  override def millSourcePath = os.pwd / os.up / "thirdparty" / "chipyard" / "generators" / "icenet"
  override def scalaVersion = "2.13.12"
  
  // Add rocket-chip as a dependency
  override def moduleDeps = Seq(
    rocketChip
  )
  
  override def ivyDeps = Agg(
    ivy"org.chipsalliance::chisel:6.5.0"
  )
  
  override def scalacPluginIvyDeps = Agg(
    ivy"org.chipsalliance:::chisel-plugin:6.5.0"
  )
}

// Define nvdla module
object nvdla extends SbtModule {
  override def millSourcePath = os.pwd / os.up / "thirdparty" / "chipyard" / "generators" / "nvdla"
  override def scalaVersion = "2.13.12"
  
  // Add rocket-chip as a dependency
  override def moduleDeps = Seq(
    rocketChip
  )
  
  override def ivyDeps = Agg(
    ivy"org.chipsalliance::chisel:6.5.0"
  )
  
  override def scalacPluginIvyDeps = Agg(
    ivy"org.chipsalliance:::chisel-plugin:6.5.0"
  )
}

// Define fftgenerator module
object fftgenerator extends SbtModule {
  override def millSourcePath = os.pwd / os.up / "thirdparty" / "chipyard" / "generators" / "fft-generator"
  override def scalaVersion = "2.13.12"
  
  // Add rocket-chip and rocket-dsp-utils as dependencies (as per build.sbt)
  override def moduleDeps = Seq(
    rocketChip,
    rocketDspUtils
  )
  
  override def ivyDeps = Agg(
    ivy"org.chipsalliance::chisel:6.5.0"
  )
  
  override def scalacPluginIvyDeps = Agg(
    ivy"org.chipsalliance:::chisel-plugin:6.5.0"
  )
}

// Define constellation module
object constellation extends SbtModule {
  override def millSourcePath = os.pwd / os.up / "thirdparty" / "chipyard" / "generators" / "constellation"
  override def scalaVersion = "2.13.12"
  
  // Add rocket-chip as a dependency
  override def moduleDeps = Seq(
    rocketChip
  )
  
  override def ivyDeps = Agg(
    ivy"org.chipsalliance::chisel:6.5.0"
  )
  
  override def scalacPluginIvyDeps = Agg(
    ivy"org.chipsalliance:::chisel-plugin:6.5.0"
  )
}

// Define boom module
object boom extends SbtModule {
  override def millSourcePath = os.pwd / os.up / "thirdparty" / "chipyard" / "generators" / "boom"
  override def scalaVersion = "2.13.12"
  
  // Add rocket-chip as a dependency
  override def moduleDeps = Seq(
    rocketChip
  )
  
  override def ivyDeps = Agg(
    ivy"org.chipsalliance::chisel:6.5.0"
  )
  
  override def scalacPluginIvyDeps = Agg(
    ivy"org.chipsalliance:::chisel-plugin:6.5.0"
  )
}

// Define tracegen module
object tracegen extends SbtModule {
  override def millSourcePath = os.pwd / os.up / "thirdparty" / "chipyard" / "generators" / "tracegen"
  override def scalaVersion = "2.13.12"
  
  // Add testchipip, rocket-chip, rocketchip_inclusive_cache, and boom as dependencies (as per build.sbt)
  override def moduleDeps = Seq(
    testchipip,
    rocketChip,
    rocketchipInclusiveCache,
    boom
  )
  
  override def ivyDeps = Agg(
    ivy"org.chipsalliance::chisel:6.5.0"
  )
  
  override def scalacPluginIvyDeps = Agg(
    ivy"org.chipsalliance:::chisel-plugin:6.5.0"
  )
}

// Define shuttle module
object shuttle extends SbtModule {
  override def millSourcePath = os.pwd / os.up / "thirdparty" / "chipyard" / "generators" / "shuttle"
  override def scalaVersion = "2.13.12"
  
  // Add rocket-chip as a dependency
  override def moduleDeps = Seq(
    rocketChip
  )
  
  override def ivyDeps = Agg(
    ivy"org.chipsalliance::chisel:6.5.0"
  )
  
  override def scalacPluginIvyDeps = Agg(
    ivy"org.chipsalliance:::chisel-plugin:6.5.0"
  )
}

// Define rocketchip_inclusive_cache module
object rocketchipInclusiveCache extends SbtModule {
  override def millSourcePath = os.pwd / os.up / "thirdparty" / "chipyard" / "generators" / "rocket-chip-inclusive-cache"
  override def scalaVersion = "2.13.12"
  
  // Override sources to match build.sbt behavior - point to design/craft directory
  override def sources = T.sources {
    super.sources() ++ Seq(PathRef(millSourcePath / "design" / "craft"))
  }
  
  // Add rocket-chip as a dependency
  override def moduleDeps = Seq(
    rocketChip
  )
  
  override def ivyDeps = Agg(
    ivy"org.chipsalliance::chisel:6.5.0"
  )
  
  override def scalacPluginIvyDeps = Agg(
    ivy"org.chipsalliance:::chisel-plugin:6.5.0"
  )
}

// Define saturn module
object saturn extends SbtModule {
  override def millSourcePath = os.pwd / os.up / "thirdparty" / "chipyard" / "generators" / "saturn"
  override def scalaVersion = "2.13.12"
  
  // Add rocket-chip and shuttle as dependencies (as per build.sbt)
  override def moduleDeps = Seq(
    rocketChip,
    shuttle
  )
  
  override def ivyDeps = Agg(
    ivy"org.chipsalliance::chisel:6.5.0"
  )
  
  override def scalacPluginIvyDeps = Agg(
    ivy"org.chipsalliance:::chisel-plugin:6.5.0"
  )
}

// Define gemmini module
object gemmini extends SbtModule {
  override def millSourcePath = os.pwd / os.up / "thirdparty" / "chipyard" / "generators" / "gemmini"
  override def scalaVersion = "2.13.12"
  
  // Add rocket-chip as a dependency
  override def moduleDeps = Seq(
    rocketChip
  )
  
  override def ivyDeps = Agg(
    ivy"org.chipsalliance::chisel:6.5.0"
  )
  
  override def scalacPluginIvyDeps = Agg(
    ivy"org.chipsalliance:::chisel-plugin:6.5.0"
  )
}

// Define sodor module
object sodor extends SbtModule {
  override def millSourcePath = os.pwd / os.up / "thirdparty" / "chipyard" / "generators" / "riscv-sodor"
  override def scalaVersion = "2.13.12"
  
  // Add rocket-chip as a dependency
  override def moduleDeps = Seq(
    rocketChip
  )
  
  override def ivyDeps = Agg(
    ivy"org.chipsalliance::chisel:6.5.0"
  )
  
  override def scalacPluginIvyDeps = Agg(
    ivy"org.chipsalliance:::chisel-plugin:6.5.0"
  )
}

// Define vexiiriscv module
object vexiiriscv extends SbtModule {
  override def millSourcePath = os.pwd / os.up / "thirdparty" / "chipyard" / "generators" / "vexiiriscv"
  override def scalaVersion = "2.13.12"
  
  // Add rocket-chip as a dependency
  override def moduleDeps = Seq(
    rocketChip
  )
  
  override def ivyDeps = Agg(
    ivy"org.chipsalliance::chisel:6.5.0"
  )
  
  override def scalacPluginIvyDeps = Agg(
    ivy"org.chipsalliance:::chisel-plugin:6.5.0"
  )
}

// Define ibex module
object ibex extends SbtModule {
  override def millSourcePath = os.pwd / os.up / "thirdparty" / "chipyard" / "generators" / "ibex"
  override def scalaVersion = "2.13.12"
  
  // Add rocket-chip as a dependency
  override def moduleDeps = Seq(
    rocketChip
  )
  
  override def ivyDeps = Agg(
    ivy"org.chipsalliance::chisel:6.5.0"
  )
  
  override def scalacPluginIvyDeps = Agg(
    ivy"org.chipsalliance:::chisel-plugin:6.5.0"
  )
}

// Define cva6 module
object cva6 extends SbtModule {
  override def millSourcePath = os.pwd / os.up / "thirdparty" / "chipyard" / "generators" / "cva6"
  override def scalaVersion = "2.13.12"
  
  // Add rocket-chip as a dependency
  override def moduleDeps = Seq(
    rocketChip
  )
  
  override def ivyDeps = Agg(
    ivy"org.chipsalliance::chisel:6.5.0"
  )
  
  override def scalacPluginIvyDeps = Agg(
    ivy"org.chipsalliance:::chisel-plugin:6.5.0"
  )
}

// Define ara module
object ara extends SbtModule {
  override def millSourcePath = os.pwd / os.up / "thirdparty" / "chipyard" / "generators" / "ara"
  override def scalaVersion = "2.13.12"
  
  // Add rocket-chip and shuttle as dependencies (as per build.sbt)
  override def moduleDeps = Seq(
    rocketChip,
    shuttle
  )
  
  override def ivyDeps = Agg(
    ivy"org.chipsalliance::chisel:6.5.0"
  )
  
  override def scalacPluginIvyDeps = Agg(
    ivy"org.chipsalliance:::chisel-plugin:6.5.0"
  )
}

// Define rerocc module
object rerocc extends SbtModule {
  override def millSourcePath = os.pwd / os.up / "thirdparty" / "chipyard" / "generators" / "rerocc"
  override def scalaVersion = "2.13.12"
  
  // Add rocket-chip, constellation, boom, and shuttle as dependencies (as per build.sbt)
  override def moduleDeps = Seq(
    rocketChip,
    constellation,
    boom,
    shuttle
  )
  
  override def ivyDeps = Agg(
    ivy"org.chipsalliance::chisel:6.5.0"
  )
  
  override def scalacPluginIvyDeps = Agg(
    ivy"org.chipsalliance:::chisel-plugin:6.5.0"
  )
}

// Define rocket-dsp-utils module
object rocketDspUtils extends SbtModule {
  override def millSourcePath = os.pwd / os.up / "thirdparty" / "chipyard" / "tools" / "rocket-dsp-utils"
  override def scalaVersion = "2.13.12"
  
  // Add rocket-chip, cde, and dsptools as dependencies (as per build.sbt)
  override def moduleDeps = Seq(
    rocketChip,
    cde,
    dsptools
  )
  
  override def ivyDeps = Agg(
    ivy"org.chipsalliance::chisel:6.5.0"
  )
  
  override def scalacPluginIvyDeps = Agg(
    ivy"org.chipsalliance:::chisel-plugin:6.5.0"
  )
}

// Define dsptools module
object dsptools extends SbtModule {
  override def millSourcePath = os.pwd / os.up / "thirdparty" / "chipyard" / "tools" / "dsptools"
  override def scalaVersion = "2.13.12"
  
  // Add rocket-chip and fixedpoint as dependencies (as per build.sbt)
  override def moduleDeps = Seq(
    rocketChip,
    fixedpoint
  )
  
  override def ivyDeps = Agg(
    ivy"org.chipsalliance::chisel:6.5.0",
    ivy"org.typelevel::spire:0.18.0",
    ivy"org.scalanlp::breeze:2.1.0",
    ivy"edu.berkeley.cs::chiseltest:6.0.0"
  )
  
  override def scalacPluginIvyDeps = Agg(
    ivy"org.chipsalliance:::chisel-plugin:6.5.0"
  )
}

// Define fixedpoint module
object fixedpoint extends SbtModule {
  override def millSourcePath = os.pwd / os.up / "thirdparty" / "chipyard" / "tools" / "fixedpoint"
  override def scalaVersion = "2.13.12"
  
  // Add rocket-chip as a dependency
  override def moduleDeps = Seq(
    rocketChip
  )
  
  override def ivyDeps = Agg(
    ivy"org.chipsalliance::chisel:6.5.0"
  )
  
  override def scalacPluginIvyDeps = Agg(
    ivy"org.chipsalliance:::chisel-plugin:6.5.0"
  )
}

// Define compressacc module
object compressacc extends SbtModule {
  override def millSourcePath = os.pwd / os.up / "thirdparty" / "chipyard" / "generators" / "compress-acc"
  override def scalaVersion = "2.13.12"
  
  // Add rocket-chip as a dependency
  override def moduleDeps = Seq(
    rocketChip
  )
  
  override def ivyDeps = Agg(
    ivy"org.chipsalliance::chisel:6.5.0"
  )
  
  override def scalacPluginIvyDeps = Agg(
    ivy"org.chipsalliance:::chisel-plugin:6.5.0"
  )
}

// Define mempress module
object mempress extends SbtModule {
  override def millSourcePath = os.pwd / os.up / "thirdparty" / "chipyard" / "generators" / "mempress"
  override def scalaVersion = "2.13.12"
  
  // Add rocket-chip as a dependency
  override def moduleDeps = Seq(
    rocketChip
  )
  
  override def ivyDeps = Agg(
    ivy"org.chipsalliance::chisel:6.5.0"
  )
  
  override def scalacPluginIvyDeps = Agg(
    ivy"org.chipsalliance:::chisel-plugin:6.5.0"
  )
}

// Define barf module
object barf extends SbtModule {
  override def millSourcePath = os.pwd / os.up / "thirdparty" / "chipyard" / "generators" / "bar-fetchers"
  override def scalaVersion = "2.13.12"
  
  // Add rocket-chip as a dependency
  override def moduleDeps = Seq(
    rocketChip
  )
  
  override def ivyDeps = Agg(
    ivy"org.chipsalliance::chisel:6.5.0"
  )
  
  override def scalacPluginIvyDeps = Agg(
    ivy"org.chipsalliance:::chisel-plugin:6.5.0"
  )
}

// Define caliptra_aes module
object caliptraAes extends SbtModule {
  override def millSourcePath = os.pwd / os.up / "thirdparty" / "chipyard" / "generators" / "caliptra-aes-acc"
  override def scalaVersion = "2.13.12"
  
  // Add rocket-chip, rocc_acc_utils, and testchipip as dependencies (as per build.sbt)
  override def moduleDeps = Seq(
    rocketChip,
    roccAccUtils,
    testchipip
  )
  
  override def ivyDeps = Agg(
    ivy"org.chipsalliance::chisel:6.5.0"
  )
  
  override def scalacPluginIvyDeps = Agg(
    ivy"org.chipsalliance:::chisel-plugin:6.5.0"
  )
}

// Define rocc_acc_utils module
object roccAccUtils extends SbtModule {
  override def millSourcePath = os.pwd / os.up / "thirdparty" / "chipyard" / "generators" / "rocc-acc-utils"
  override def scalaVersion = "2.13.12"
  
  // Add rocket-chip as a dependency
  override def moduleDeps = Seq(
    rocketChip
  )
  
  override def ivyDeps = Agg(
    ivy"org.chipsalliance::chisel:6.5.0"
  )
  
  override def scalacPluginIvyDeps = Agg(
    ivy"org.chipsalliance:::chisel-plugin:6.5.0"
  )
}

// Define firrtl2 module
object firrtl2 extends SbtModule {
  override def millSourcePath = os.pwd / os.up / "thirdparty" / "chipyard" / "tools" / "firrtl2"
  override def scalaVersion = "2.13.12"
  
  // Override sources to include generated ANTLR sources and BuildInfo
  override def sources = T.sources {
    val baseSources = super.sources()
    // Include pre-generated sources from target directory
    val generatedDir = millSourcePath / "src" / "target" / "scala-2.13" / "src_managed" / "main"
    if (os.exists(generatedDir)) {
      baseSources ++ Seq(PathRef(generatedDir))
    } else {
      baseSources
    }
  }
  
  override def ivyDeps = Agg(
    ivy"org.chipsalliance::chisel:6.5.0",
    ivy"org.scalatest::scalatest:3.2.14",
    ivy"org.scalatestplus::scalacheck-1-15:3.2.11.0",
    ivy"com.github.scopt::scopt:4.1.0",
    ivy"org.json4s::json4s-native:4.1.0-M4",
    ivy"org.apache.commons:commons-text:1.10.0",
    ivy"com.lihaoyi::os-lib:0.8.1",
    ivy"org.scala-lang.modules::scala-parallel-collections:1.0.4",
    ivy"org.antlr:antlr4-runtime:4.9.3"
  )
  
  override def scalacPluginIvyDeps = Agg(
    ivy"org.chipsalliance:::chisel-plugin:6.5.0"
  )
  
  override def scalacOptions = Seq(
    "-language:reflectiveCalls",
    "-language:existentials",
    "-language:implicitConversions"
  )
}

// Define firrtl2_bridge module
object firrtl2Bridge extends SbtModule {
  override def millSourcePath = os.pwd / os.up / "thirdparty" / "chipyard" / "tools" / "firrtl2" / "bridge"
  override def scalaVersion = "2.13.12"
  
  // Add firrtl2 as a dependency
  override def moduleDeps = Seq(
    firrtl2
  )
  
  override def ivyDeps = Agg(
    ivy"org.chipsalliance::chisel:6.5.0"
  )
  
  override def scalacPluginIvyDeps = Agg(
    ivy"org.chipsalliance:::chisel-plugin:6.5.0"
  )
}