import mill._
import $file.fetch.{build => fetch_build}

// Memory-level parallelism: what lets a CPU keep memory accesses in flight
// instead of waiting on each one (caches, line buffers, prefetch, non-blocking
// misses). A CPU runs without it, one access at a time.
def sources = T.sources(
  fetch_build.sources()
)

def emitSources = T.sources(
  fetch_build.emitSources()
)
