import mill._
import $file.acc.{build => acc_build}
import $file.ant.{build => ant_build}
import $file.tbus.{build => tbus_build}
import $file.tlink.{build => tlink_build}
import $file.sims.{build => sims_build}

def sources = T.sources(
  acc_build.sources() ++
  Seq(
    PathRef(millSourcePath / os.RelPath("configs/ThreadParam.scala"))
  ) ++
  ant_build.sources() ++
  tbus_build.sources() ++
  tlink_build.sources()
)

// The virtual environment of a thread alone, on the simulation layer.
def simSources = T.sources(
  sims_build.sources()
)

def emitSources = T.sources(
  acc_build.emitSources() ++
  ant_build.emitSources() ++
  tlink_build.emitSources()
)
