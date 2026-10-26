import mill._
import $file.access.{build => access_build}
import $file.home.{build => home_build}

def sources = T.sources(
  Seq(
    PathRef(millSourcePath / os.RelPath("Memory.scala")),
    PathRef(millSourcePath / os.RelPath("MemorySystem.scala")),
    PathRef(millSourcePath / os.RelPath("ThreadMemory.scala")),
    PathRef(millSourcePath / os.RelPath("configs/MemoryParam.scala")),
    PathRef(millSourcePath / os.RelPath("uncached/Uncached.scala")),
    PathRef(millSourcePath / os.RelPath("uncached/configs/UncachedParam.scala"))
  ) ++
  access_build.sources() ++
  home_build.sources()
)

def emitSources = T.sources(
  access_build.emitSources() ++
  home_build.emitSources()
)
