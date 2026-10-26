package framework.core.shm

import framework.acc.frontend.configs.FrontendParam
import framework.acc.memdomain.configs.MemDomainParam
import framework.core.tbus.configs.{Params => TBusParam}

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.ip.memory.mesh_shm.{MeshSharedMem, MeshSharedMemIO}

@instantiable
class SHM(frontendParam: FrontendParam, memParam: MemDomainParam, tbusParam: TBusParam) extends Module {
  val p  = tbusParam.router.mesh(frontendParam, memParam, tbusParam)
  @public
  val io = IO(new MeshSharedMemIO(p))

  val memory: Instance[MeshSharedMem] = Instantiate(new MeshSharedMem(p))
  io <> memory.io
}
