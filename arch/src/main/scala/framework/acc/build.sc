import mill._
import $file.debug.{build => debug_build}
import $file.balldomain.{build => balldomain_build}
import $file.frontend.{build => frontend_build}
import $file.gpdomain.{build => gpdomain_build}
import $file.memdomain.{build => memdomain_build}

def sources = T.sources(
  Seq(
    PathRef(millSourcePath / os.RelPath("configs/AccParam.scala"))
  ) ++
  debug_build.sources() ++
  balldomain_build.sources() ++
  frontend_build.sources() ++
  gpdomain_build.sources() ++
  memdomain_build.sources()
)

def emitSources = T.sources(
  debug_build.emitSources() ++
  balldomain_build.emitSources() ++
  frontend_build.emitSources() ++
  gpdomain_build.emitSources() ++
  memdomain_build.emitSources()
)
