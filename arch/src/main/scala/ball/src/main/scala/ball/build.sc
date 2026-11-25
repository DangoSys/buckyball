import mill._
import $file.blink.{build => blink_build}

def sources = T.sources(
  blink_build.sources()
)

def emitSources = T.sources(
  blink_build.emitSources()
)
