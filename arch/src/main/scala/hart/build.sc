import mill._
import $file.cbus.{build => cbus_build}
import $file.hlink.{build => hlink_build}
import $file.execution.{build => execution_build}
import $file.memory.{build => memory_build}
import $file.sims.{build => sims_build}

def sources = T.sources(
  hlink_build.sources() ++
  execution_build.sources() ++
  memory_build.sources()
)

// The simulation tops assemble every abstraction: a layer of their own. A
// hart's whole configuration and the bus that instantiates harts from it are
// assembly too: they name every abstraction a hart may compose.
def simSources = T.sources(
  sims_build.sources() ++
  cbus_build.sources() ++
  Seq(
    PathRef(millSourcePath / os.RelPath("configs/HartParam.scala"))
  )
)

def emitSources = T.sources(
  hlink_build.emitSources() ++
  execution_build.emitSources() ++
  memory_build.emitSources()
)
