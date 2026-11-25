import mill._
import $file.controller.{build => controller_build}
import $file.shm.{build => shm_build}
import $file.tbus.{build => tbus_build}
import $file.tss.{build => tss_build}

// Thread-level parallelism: what lets a hart run its threads of work in
// parallel: the controller that schedules their tasks, the task storage, the
// thread bus and the shared memory between them, and the threads' DMA. A hart
// runs without it, a single thread.
def sources = T.sources(
  Seq(
    PathRef(millSourcePath / os.RelPath("memory/Dataflow.scala")),
    PathRef(millSourcePath / os.RelPath("memory/DmaMemory.scala")),
    PathRef(millSourcePath / os.RelPath("memory/DmaRoute.scala"))
  ) ++
  controller_build.sources() ++
  shm_build.sources() ++
  tbus_build.sources() ++
  tss_build.sources()
)

def emitSources = T.sources(
  controller_build.emitSources() ++
  shm_build.emitSources() ++
  tbus_build.emitSources() ++
  tss_build.emitSources()
)
