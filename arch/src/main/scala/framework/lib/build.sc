import mill._
import $file.arbiter.{build => arbiter_build}
import $file.arith.{build => arith_build}
import $file.dpi.{build => dpi_build}
import $file.hash.{build => hash_build}
import $file.queue.{build => queue_build}

def sources = T.sources(
  arbiter_build.sources() ++
  arith_build.sources() ++
  dpi_build.sources() ++
  hash_build.sources() ++
  queue_build.sources()
)

def emitSources = T.sources(
  arbiter_build.emitSources() ++
  arith_build.emitSources() ++
  dpi_build.emitSources() ++
  hash_build.emitSources() ++
  queue_build.emitSources()
)
