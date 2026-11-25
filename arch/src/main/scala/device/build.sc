import mill._
import $file.dlink.{build => dlink_build}
import $file.peripherals.{build => peripherals_build}
import $file.shm.{build => shm_build}
import $file.sims.{build => sims_build}

def sources = T.sources(
  Seq(
    PathRef(millSourcePath / os.RelPath("configs/DeviceParam.scala"))
  ) ++
  dlink_build.sources() ++
  peripherals_build.sources() ++
  shm_build.sources()
)

// The simulation tops assemble every abstraction: a layer of their own.
def simSources = T.sources(
  sims_build.sources()
)

def emitSources = T.sources(
  dlink_build.emitSources() ++
  peripherals_build.emitSources() ++
  shm_build.emitSources()
)
