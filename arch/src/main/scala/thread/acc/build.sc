import mill._
import $file.debug.{build => debug_build}
import $file.frontend.{build => frontend_build}
import $file.gpdomain.{build => gpdomain_build}
import $file.memdomain.{build => memdomain_build}

def sources = T.sources(
  Seq(
    PathRef(millSourcePath / os.RelPath("configs/AccParam.scala")),
    PathRef(millSourcePath / os.RelPath("configs/BallLink.scala")),
    PathRef(millSourcePath / os.RelPath("frontend/configs/FrontendParam.scala")),
    PathRef(millSourcePath / os.RelPath("gpdomain/configs/GpDomainParam.scala")),
    PathRef(millSourcePath / os.RelPath("gpdomain/execution/configs/Params.scala")),
    PathRef(millSourcePath / os.RelPath("gpdomain/loader/configs/Params.scala")),
    PathRef(millSourcePath / os.RelPath("gpdomain/memory/configs/Params.scala")),
    PathRef(millSourcePath / os.RelPath("memdomain/Configer.scala")),
    PathRef(millSourcePath / os.RelPath("memdomain/Layout.scala")),
    PathRef(millSourcePath / os.RelPath("memdomain/bank_router/BankRouter.scala")),
    PathRef(millSourcePath / os.RelPath("memdomain/configs/MemDomainParam.scala")),
    PathRef(millSourcePath / os.RelPath("memdomain/toddr/DmaPort.scala")),
    PathRef(millSourcePath / os.RelPath("memdomain/tomem/configs/Params.scala")),
    PathRef(millSourcePath / os.RelPath("shm/Banks.scala"))
  ) ++
  debug_build.sources() ++
  frontend_build.sources() ++
  gpdomain_build.sources() ++
  memdomain_build.sources()
)

def emitSources = T.sources(
  debug_build.emitSources() ++
  frontend_build.emitSources() ++
  gpdomain_build.emitSources() ++
  memdomain_build.emitSources()
)
