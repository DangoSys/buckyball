import mill._
import $file.cbus.{build => cbus_build}
import $file.ddr.{build => ddr_build}
import $file.dlink.{build => dlink_build}
import $file.peripherals.{build => peripherals_build}
import $file.shm.{build => shm_build}

def sources = T.sources(
  Seq(
    PathRef(millSourcePath / os.RelPath("configs/DeviceParam.scala"))
  ) ++
  cbus_build.sources() ++
  ddr_build.sources() ++
  dlink_build.sources() ++
  peripherals_build.sources() ++
  shm_build.sources()
)

def emitSources = T.sources(
  cbus_build.emitSources() ++
  ddr_build.emitSources() ++
  dlink_build.emitSources() ++
  peripherals_build.emitSources() ++
  shm_build.emitSources()
)
