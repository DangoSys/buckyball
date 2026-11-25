package framework.tlp.shm

import framework.tlp.tbus.configs.{Params => TBusParam}

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.tlp.shm.{MeshSharedMem, MeshSharedMemIO}
import framework.generic.acc.frontend.configs.FrontendParam
import framework.generic.acc.memdomain.configs.MemDomainParam

@instantiable
class SHM(frontendParam: FrontendParam, memParam: MemDomainParam, tbusParam: TBusParam) extends Module {
  val p  = tbusParam.router.mesh(frontendParam, memParam, tbusParam)
  @public
  val io = IO(new MeshSharedMemIO(p))

  val memory: Instance[MeshSharedMem] = Instantiate(new MeshSharedMem(p))
  io <> memory.io
}
