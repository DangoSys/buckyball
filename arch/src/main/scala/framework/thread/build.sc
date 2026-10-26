import mill._
import $file.ant.{build => ant_build}
import $file.tlink.{build => tlink_build}

def sources = T.sources(
  Seq(
    PathRef(millSourcePath / os.RelPath("configs/ThreadParam.scala"))
  ) ++
  ant_build.sources() ++
  tlink_build.sources()
)

def emitSources = T.sources(
  ant_build.emitSources() ++
  tlink_build.emitSources()
)
