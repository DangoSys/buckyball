import mill._
import $file.clink.{build => clink_build}
import $file.controller.{build => controller_build}
import $file.execution.{build => execution_build}
import $file.memory.{build => memory_build}
import $file.shm.{build => shm_build}
import $file.tbus.{build => tbus_build}
import $file.tss.{build => tss_build}

def sources = T.sources(
  Seq(
    PathRef(millSourcePath / os.RelPath("configs/CoreParam.scala"))
  ) ++
  clink_build.sources() ++
  controller_build.sources() ++
  execution_build.sources() ++
  memory_build.sources() ++
  shm_build.sources() ++
  tbus_build.sources() ++
  tss_build.sources()
)

def emitSources = T.sources(
  clink_build.emitSources() ++
  controller_build.emitSources() ++
  execution_build.emitSources() ++
  memory_build.emitSources() ++
  shm_build.emitSources() ++
  tbus_build.emitSources() ++
  tss_build.emitSources()
)
