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
    chipyardModule,
    chipyardGeneratorsModule
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
  
  object test extends SbtTests with TestModule.ScalaTest {
    override def ivyDeps = m.ivyDeps() ++ Agg(
      ivy"org.scalatest::scalatest::3.2.19"
      // ivy"org.scalatest::scalatest:3.2.16"
    )
  }
}

// Define cde module - must be compiled first
object cdeModule extends SbtModule {
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
object hardfloatModule extends SbtModule {
  override def millSourcePath = os.pwd / os.up / "thirdparty" / "chipyard" / "generators" / "hardfloat"
  override def scalaVersion = "2.13.12"
  
  // Add cde dependency
  override def moduleDeps = Seq(
    cdeModule
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
object midasTargetUtilsModule extends SbtModule {
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
object diplomacyModule extends SbtModule {
  override def millSourcePath = os.pwd / os.up / "thirdparty" / "chipyard" / "generators" / "diplomacy" / "diplomacy"
  override def scalaVersion = "2.13.12"
  
  // Add cde dependency first
  override def moduleDeps = Seq(
    cdeModule
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
object rocketChipModule extends SbtModule {
  override def millSourcePath = os.pwd / os.up / "thirdparty" / "chipyard" / "generators" / "rocket-chip"
  override def scalaVersion = "2.13.12"
  
  // Add required dependencies for rocket-chip
  override def moduleDeps = Seq(
    diplomacyModule,
    cdeModule,
    hardfloatModule,
    midasTargetUtilsModule
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
object chipyardModule extends SbtModule {
  override def millSourcePath = os.pwd / os.up / "thirdparty" / "chipyard"
  override def scalaVersion = "2.13.12"
  
  // Override sources to include tools/stage and generators/chipyard directories (as per build.sbt)
  override def sources = T.sources {
    super.sources() ++ Seq(
      PathRef(millSourcePath / "tools" / "stage" / "src" / "main" / "scala"),
      PathRef(millSourcePath / "generators" / "chipyard" / "src" / "main" / "scala")
    )
  }
  
  // Add all required dependencies as per build.sbt
  override def moduleDeps = Seq(
    testchipipModule,
    rocketChipModule,
    boomModule,
    rocketChipBlocksModule,
    rocketchipInclusiveCacheModule,
    dsptoolsModule,
    rocketDspUtilsModule,
    gemminiModule,
    icenetModule,
    tracegenModule,
    cva6Module,
    nvdlaModule,
    sodorModule,
    ibexModule,
    fftgeneratorModule,
    constellationModule,
    mempressModule,
    barfModule,
    shuttleModule,
    caliptraAesModule,
    reroccModule,
    compressaccModule,
    saturnModule,
    araModule,
    firrtl2BridgeModule,
    vexiiriscvModule
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
object chipyardGeneratorsModule extends SbtModule {
  override def millSourcePath = os.pwd / os.up / "thirdparty" / "chipyard" / "generators" / "chipyard"
  override def scalaVersion = "2.13.12"
  
  // Add chipyard, testchipip, and other required modules as dependencies
  override def moduleDeps = Seq(
    chipyardModule,
    testchipipModule,
    icenetModule,
    nvdlaModule,
    fftgeneratorModule,
    constellationModule,
    reroccModule,
    rocketDspUtilsModule,
    dsptoolsModule,
    fixedpointModule,
    tracegenModule,
    boomModule,
    shuttleModule,
    araModule,
    saturnModule,
    gemminiModule,
    sodorModule,
    compressaccModule,
    mempressModule,
    barfModule,
    caliptraAesModule,
    roccAccUtilsModule,
    vexiiriscvModule,
    ibexModule,
    cva6Module
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
object testchipipModule extends SbtModule {
  override def millSourcePath = os.pwd / os.up / "thirdparty" / "chipyard" / "generators" / "testchipip"
  override def scalaVersion = "2.13.12"
  
  // Add rocket-chip and rocket-chip-blocks as dependencies
  override def moduleDeps = Seq(
    rocketChipModule,
    rocketChipBlocksModule
  )
  
  override def ivyDeps = Agg(
    ivy"org.chipsalliance::chisel:6.5.0"
  )
  
  override def scalacPluginIvyDeps = Agg(
    ivy"org.chipsalliance:::chisel-plugin:6.5.0"
  )
}

// Define rocket-chip-blocks module (contains sifive package)
object rocketChipBlocksModule extends SbtModule {
  override def millSourcePath = os.pwd / os.up / "thirdparty" / "chipyard" / "generators" / "rocket-chip-blocks"
  override def scalaVersion = "2.13.12"
  
  // Add rocket-chip as a dependency
  override def moduleDeps = Seq(
    rocketChipModule
  )
  
  override def ivyDeps = Agg(
    ivy"org.chipsalliance::chisel:6.5.0"
  )
  
  override def scalacPluginIvyDeps = Agg(
    ivy"org.chipsalliance:::chisel-plugin:6.5.0"
  )
}

// Define icenet module
object icenetModule extends SbtModule {
  override def millSourcePath = os.pwd / os.up / "thirdparty" / "chipyard" / "generators" / "icenet"
  override def scalaVersion = "2.13.12"
  
  // Add rocket-chip as a dependency
  override def moduleDeps = Seq(
    rocketChipModule
  )
  
  override def ivyDeps = Agg(
    ivy"org.chipsalliance::chisel:6.5.0"
  )
  
  override def scalacPluginIvyDeps = Agg(
    ivy"org.chipsalliance:::chisel-plugin:6.5.0"
  )
}

// Define nvdla module
object nvdlaModule extends SbtModule {
  override def millSourcePath = os.pwd / os.up / "thirdparty" / "chipyard" / "generators" / "nvdla"
  override def scalaVersion = "2.13.12"
  
  // Add rocket-chip as a dependency
  override def moduleDeps = Seq(
    rocketChipModule
  )
  
  override def ivyDeps = Agg(
    ivy"org.chipsalliance::chisel:6.5.0"
  )
  
  override def scalacPluginIvyDeps = Agg(
    ivy"org.chipsalliance:::chisel-plugin:6.5.0"
  )
}

// Define fftgenerator module
object fftgeneratorModule extends SbtModule {
  override def millSourcePath = os.pwd / os.up / "thirdparty" / "chipyard" / "generators" / "fft-generator"
  override def scalaVersion = "2.13.12"
  
  // Add rocket-chip and rocket-dsp-utils as dependencies (as per build.sbt)
  override def moduleDeps = Seq(
    rocketChipModule,
    rocketDspUtilsModule
  )
  
  override def ivyDeps = Agg(
    ivy"org.chipsalliance::chisel:6.5.0"
  )
  
  override def scalacPluginIvyDeps = Agg(
    ivy"org.chipsalliance:::chisel-plugin:6.5.0"
  )
}

// Define constellation module
object constellationModule extends SbtModule {
  override def millSourcePath = os.pwd / os.up / "thirdparty" / "chipyard" / "generators" / "constellation"
  override def scalaVersion = "2.13.12"
  
  // Add rocket-chip as a dependency
  override def moduleDeps = Seq(
    rocketChipModule
  )
  
  override def ivyDeps = Agg(
    ivy"org.chipsalliance::chisel:6.5.0"
  )
  
  override def scalacPluginIvyDeps = Agg(
    ivy"org.chipsalliance:::chisel-plugin:6.5.0"
  )
}

// Define boom module
object boomModule extends SbtModule {
  override def millSourcePath = os.pwd / os.up / "thirdparty" / "chipyard" / "generators" / "boom"
  override def scalaVersion = "2.13.12"
  
  // Add rocket-chip as a dependency
  override def moduleDeps = Seq(
    rocketChipModule
  )
  
  override def ivyDeps = Agg(
    ivy"org.chipsalliance::chisel:6.5.0"
  )
  
  override def scalacPluginIvyDeps = Agg(
    ivy"org.chipsalliance:::chisel-plugin:6.5.0"
  )
}

// Define tracegen module
object tracegenModule extends SbtModule {
  override def millSourcePath = os.pwd / os.up / "thirdparty" / "chipyard" / "generators" / "tracegen"
  override def scalaVersion = "2.13.12"
  
  // Add testchipip, rocket-chip, rocketchip_inclusive_cache, and boom as dependencies (as per build.sbt)
  override def moduleDeps = Seq(
    testchipipModule,
    rocketChipModule,
    rocketchipInclusiveCacheModule,
    boomModule
  )
  
  override def ivyDeps = Agg(
    ivy"org.chipsalliance::chisel:6.5.0"
  )
  
  override def scalacPluginIvyDeps = Agg(
    ivy"org.chipsalliance:::chisel-plugin:6.5.0"
  )
}

// Define shuttle module
object shuttleModule extends SbtModule {
  override def millSourcePath = os.pwd / os.up / "thirdparty" / "chipyard" / "generators" / "shuttle"
  override def scalaVersion = "2.13.12"
  
  // Add rocket-chip as a dependency
  override def moduleDeps = Seq(
    rocketChipModule
  )
  
  override def ivyDeps = Agg(
    ivy"org.chipsalliance::chisel:6.5.0"
  )
  
  override def scalacPluginIvyDeps = Agg(
    ivy"org.chipsalliance:::chisel-plugin:6.5.0"
  )
}

// Define rocketchip_inclusive_cache module
object rocketchipInclusiveCacheModule extends SbtModule {
  override def millSourcePath = os.pwd / os.up / "thirdparty" / "chipyard" / "generators" / "rocket-chip-inclusive-cache"
  override def scalaVersion = "2.13.12"
  
  // Override sources to match build.sbt behavior - point to design/craft directory
  override def sources = T.sources {
    super.sources() ++ Seq(PathRef(millSourcePath / "design" / "craft"))
  }
  
  // Add rocket-chip as a dependency
  override def moduleDeps = Seq(
    rocketChipModule
  )
  
  override def ivyDeps = Agg(
    ivy"org.chipsalliance::chisel:6.5.0"
  )
  
  override def scalacPluginIvyDeps = Agg(
    ivy"org.chipsalliance:::chisel-plugin:6.5.0"
  )
}

// Define saturn module
object saturnModule extends SbtModule {
  override def millSourcePath = os.pwd / os.up / "thirdparty" / "chipyard" / "generators" / "saturn"
  override def scalaVersion = "2.13.12"
  
  // Add rocket-chip and shuttle as dependencies (as per build.sbt)
  override def moduleDeps = Seq(
    rocketChipModule,
    shuttleModule
  )
  
  override def ivyDeps = Agg(
    ivy"org.chipsalliance::chisel:6.5.0"
  )
  
  override def scalacPluginIvyDeps = Agg(
    ivy"org.chipsalliance:::chisel-plugin:6.5.0"
  )
}

// Define gemmini module
object gemminiModule extends SbtModule {
  override def millSourcePath = os.pwd / os.up / "thirdparty" / "chipyard" / "generators" / "gemmini"
  override def scalaVersion = "2.13.12"
  
  // Add rocket-chip as a dependency
  override def moduleDeps = Seq(
    rocketChipModule
  )
  
  override def ivyDeps = Agg(
    ivy"org.chipsalliance::chisel:6.5.0"
  )
  
  override def scalacPluginIvyDeps = Agg(
    ivy"org.chipsalliance:::chisel-plugin:6.5.0"
  )
}

// Define sodor module
object sodorModule extends SbtModule {
  override def millSourcePath = os.pwd / os.up / "thirdparty" / "chipyard" / "generators" / "riscv-sodor"
  override def scalaVersion = "2.13.12"
  
  // Add rocket-chip as a dependency
  override def moduleDeps = Seq(
    rocketChipModule
  )
  
  override def ivyDeps = Agg(
    ivy"org.chipsalliance::chisel:6.5.0"
  )
  
  override def scalacPluginIvyDeps = Agg(
    ivy"org.chipsalliance:::chisel-plugin:6.5.0"
  )
}

// Define vexiiriscv module
object vexiiriscvModule extends SbtModule {
  override def millSourcePath = os.pwd / os.up / "thirdparty" / "chipyard" / "generators" / "vexiiriscv"
  override def scalaVersion = "2.13.12"
  
  // Add rocket-chip as a dependency
  override def moduleDeps = Seq(
    rocketChipModule
  )
  
  override def ivyDeps = Agg(
    ivy"org.chipsalliance::chisel:6.5.0"
  )
  
  override def scalacPluginIvyDeps = Agg(
    ivy"org.chipsalliance:::chisel-plugin:6.5.0"
  )
}

// Define ibex module
object ibexModule extends SbtModule {
  override def millSourcePath = os.pwd / os.up / "thirdparty" / "chipyard" / "generators" / "ibex"
  override def scalaVersion = "2.13.12"
  
  // Add rocket-chip as a dependency
  override def moduleDeps = Seq(
    rocketChipModule
  )
  
  override def ivyDeps = Agg(
    ivy"org.chipsalliance::chisel:6.5.0"
  )
  
  override def scalacPluginIvyDeps = Agg(
    ivy"org.chipsalliance:::chisel-plugin:6.5.0"
  )
}

// Define cva6 module
object cva6Module extends SbtModule {
  override def millSourcePath = os.pwd / os.up / "thirdparty" / "chipyard" / "generators" / "cva6"
  override def scalaVersion = "2.13.12"
  
  // Add rocket-chip as a dependency
  override def moduleDeps = Seq(
    rocketChipModule
  )
  
  override def ivyDeps = Agg(
    ivy"org.chipsalliance::chisel:6.5.0"
  )
  
  override def scalacPluginIvyDeps = Agg(
    ivy"org.chipsalliance:::chisel-plugin:6.5.0"
  )
}

// Define ara module
object araModule extends SbtModule {
  override def millSourcePath = os.pwd / os.up / "thirdparty" / "chipyard" / "generators" / "ara"
  override def scalaVersion = "2.13.12"
  
  // Add rocket-chip and shuttle as dependencies (as per build.sbt)
  override def moduleDeps = Seq(
    rocketChipModule,
    shuttleModule
  )
  
  override def ivyDeps = Agg(
    ivy"org.chipsalliance::chisel:6.5.0"
  )
  
  override def scalacPluginIvyDeps = Agg(
    ivy"org.chipsalliance:::chisel-plugin:6.5.0"
  )
}

// Define rerocc module
object reroccModule extends SbtModule {
  override def millSourcePath = os.pwd / os.up / "thirdparty" / "chipyard" / "generators" / "rerocc"
  override def scalaVersion = "2.13.12"
  
  // Add rocket-chip, constellation, boom, and shuttle as dependencies (as per build.sbt)
  override def moduleDeps = Seq(
    rocketChipModule,
    constellationModule,
    boomModule,
    shuttleModule
  )
  
  override def ivyDeps = Agg(
    ivy"org.chipsalliance::chisel:6.5.0"
  )
  
  override def scalacPluginIvyDeps = Agg(
    ivy"org.chipsalliance:::chisel-plugin:6.5.0"
  )
}

// Define rocket-dsp-utils module
object rocketDspUtilsModule extends SbtModule {
  override def millSourcePath = os.pwd / os.up / "thirdparty" / "chipyard" / "tools" / "rocket-dsp-utils"
  override def scalaVersion = "2.13.12"
  
  // Add rocket-chip, cde, and dsptools as dependencies (as per build.sbt)
  override def moduleDeps = Seq(
    rocketChipModule,
    cdeModule,
    dsptoolsModule
  )
  
  override def ivyDeps = Agg(
    ivy"org.chipsalliance::chisel:6.5.0"
  )
  
  override def scalacPluginIvyDeps = Agg(
    ivy"org.chipsalliance:::chisel-plugin:6.5.0"
  )
}

// Define dsptools module
object dsptoolsModule extends SbtModule {
  override def millSourcePath = os.pwd / os.up / "thirdparty" / "chipyard" / "tools" / "dsptools"
  override def scalaVersion = "2.13.12"
  
  // Add rocket-chip and fixedpoint as dependencies (as per build.sbt)
  override def moduleDeps = Seq(
    rocketChipModule,
    fixedpointModule
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
object fixedpointModule extends SbtModule {
  override def millSourcePath = os.pwd / os.up / "thirdparty" / "chipyard" / "tools" / "fixedpoint"
  override def scalaVersion = "2.13.12"
  
  // Add rocket-chip as a dependency
  override def moduleDeps = Seq(
    rocketChipModule
  )
  
  override def ivyDeps = Agg(
    ivy"org.chipsalliance::chisel:6.5.0"
  )
  
  override def scalacPluginIvyDeps = Agg(
    ivy"org.chipsalliance:::chisel-plugin:6.5.0"
  )
}

// Define compressacc module
object compressaccModule extends SbtModule {
  override def millSourcePath = os.pwd / os.up / "thirdparty" / "chipyard" / "generators" / "compress-acc"
  override def scalaVersion = "2.13.12"
  
  // Add rocket-chip as a dependency
  override def moduleDeps = Seq(
    rocketChipModule
  )
  
  override def ivyDeps = Agg(
    ivy"org.chipsalliance::chisel:6.5.0"
  )
  
  override def scalacPluginIvyDeps = Agg(
    ivy"org.chipsalliance:::chisel-plugin:6.5.0"
  )
}

// Define mempress module
object mempressModule extends SbtModule {
  override def millSourcePath = os.pwd / os.up / "thirdparty" / "chipyard" / "generators" / "mempress"
  override def scalaVersion = "2.13.12"
  
  // Add rocket-chip as a dependency
  override def moduleDeps = Seq(
    rocketChipModule
  )
  
  override def ivyDeps = Agg(
    ivy"org.chipsalliance::chisel:6.5.0"
  )
  
  override def scalacPluginIvyDeps = Agg(
    ivy"org.chipsalliance:::chisel-plugin:6.5.0"
  )
}

// Define barf module
object barfModule extends SbtModule {
  override def millSourcePath = os.pwd / os.up / "thirdparty" / "chipyard" / "generators" / "bar-fetchers"
  override def scalaVersion = "2.13.12"
  
  // Add rocket-chip as a dependency
  override def moduleDeps = Seq(
    rocketChipModule
  )
  
  override def ivyDeps = Agg(
    ivy"org.chipsalliance::chisel:6.5.0"
  )
  
  override def scalacPluginIvyDeps = Agg(
    ivy"org.chipsalliance:::chisel-plugin:6.5.0"
  )
}

// Define caliptra_aes module
object caliptraAesModule extends SbtModule {
  override def millSourcePath = os.pwd / os.up / "thirdparty" / "chipyard" / "generators" / "caliptra-aes-acc"
  override def scalaVersion = "2.13.12"
  
  // Add rocket-chip, rocc_acc_utils, and testchipip as dependencies (as per build.sbt)
  override def moduleDeps = Seq(
    rocketChipModule,
    roccAccUtilsModule,
    testchipipModule
  )
  
  override def ivyDeps = Agg(
    ivy"org.chipsalliance::chisel:6.5.0"
  )
  
  override def scalacPluginIvyDeps = Agg(
    ivy"org.chipsalliance:::chisel-plugin:6.5.0"
  )
}

// Define rocc_acc_utils module
object roccAccUtilsModule extends SbtModule {
  override def millSourcePath = os.pwd / os.up / "thirdparty" / "chipyard" / "generators" / "rocc-acc-utils"
  override def scalaVersion = "2.13.12"
  
  // Add rocket-chip as a dependency
  override def moduleDeps = Seq(
    rocketChipModule
  )
  
  override def ivyDeps = Agg(
    ivy"org.chipsalliance::chisel:6.5.0"
  )
  
  override def scalacPluginIvyDeps = Agg(
    ivy"org.chipsalliance:::chisel-plugin:6.5.0"
  )
}

// Define firrtl2 module
object firrtl2Module extends SbtModule {
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
object firrtl2BridgeModule extends SbtModule {
  override def millSourcePath = os.pwd / os.up / "thirdparty" / "chipyard" / "tools" / "firrtl2" / "bridge"
  override def scalaVersion = "2.13.12"
  
  // Add firrtl2 as a dependency
  override def moduleDeps = Seq(
    firrtl2Module
  )
  
  override def ivyDeps = Agg(
    ivy"org.chipsalliance::chisel:6.5.0"
  )
  
  override def scalacPluginIvyDeps = Agg(
    ivy"org.chipsalliance:::chisel-plugin:6.5.0"
  )
}