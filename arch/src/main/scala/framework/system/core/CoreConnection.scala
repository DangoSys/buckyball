package framework.system.core

import chisel3._
import chisel3.experimental.hierarchy.Instance
import framework.ant.LocalExecution
import framework.system.core.accelerator.{AntAdmission, BuckyballAccelerator}
import framework.system.core.clink.CLinkIO

object CoreConnection {

  def accelerator(link: AcceleratorCLink, accelerator: Instance[BuckyballAccelerator]): Unit = {
    accelerator.io.cmd <> link.npu.command
    link.npu.response <> accelerator.io.resp
    link.npu.allocation                         := accelerator.io.allocation
    link.npu.retired                            := accelerator.io.retired
    link.npu.fault                              := accelerator.io.fault
    link.npu.busy                               := !accelerator.io.idle
    link.npu.interrupt                          := accelerator.io.interrupt
    link.npu.footprints                         := accelerator.io.footprints
    link.mem <> accelerator.io.dma
    accelerator.io.hartid                       := link.hartId
    accelerator.io.sharedBankOwnerHartId        := link.shmOwner
    link.shm.requests <> accelerator.io.shared_mem_req
    link.shm.move <> accelerator.io.mvover
    accelerator.io.meshLocalBank <> link.shm.local
    link.shm.config <> accelerator.io.shared_config
    link.shm.queryValid                         := accelerator.io.shared_query_valid
    link.shm.queryVbank                         := accelerator.io.shared_query_vbank_id
    accelerator.io.shared_query_group_count     := link.shm.queryGroups
    link.shm.barrierArrive                      := accelerator.io.barrier_arrive
    accelerator.io.barrier_release              := link.shm.barrierRelease
    accelerator.io.shared_bank_hashes.foreach(_ := link.sharedHashes.get)
  }

  def ant(
    clink:       CLinkIO,
    execution:   Instance[LocalExecution],
    accelerator: Instance[BuckyballAccelerator],
    issuer:      Instance[AntAdmission],
    signature:   BigInt
  ): Unit = {
    execution.io.local <> clink.local
    clink.ctrl.fingerprint                      := signature.U(64.W)
    clink.ctrl.online                           := true.B
    issuer.io.bind.valid                        := clink.ctrl.bind.valid
    issuer.io.bind.bits.task                    := clink.ctrl.bind.bits.task
    issuer.io.bind.bits.snapshot                := clink.ctrl.bind.bits.snapshot
    issuer.io.command <> execution.io.command
    execution.io.response <> issuer.io.response
    issuer.io.cancel                            := clink.local.cancel
    issuer.io.workDrained                       := clink.ctrl.workDrained
    issuer.io.halted                            := clink.ctrl.halted
    clink.ctrl.drained                          := issuer.io.drained
    execution.io.npuDrained                     := issuer.io.drained
    clink.admission <> issuer.io.admission
    clink.memory <> issuer.io.memory
    accelerator.io.cmd <> clink.npu.command
    clink.npu.response <> accelerator.io.resp
    clink.npu.allocation                        := accelerator.io.allocation
    clink.npu.retired                           := accelerator.io.retired
    clink.npu.fault                             := accelerator.io.fault
    clink.npu.busy                              := !accelerator.io.idle
    clink.npu.interrupt                         := accelerator.io.interrupt
    clink.npu.footprints                        := accelerator.io.footprints
    clink.mem <> accelerator.io.dma
    accelerator.io.hartid                       := clink.ctrl.hartId
    accelerator.io.sharedBankOwnerHartId        := clink.ctrl.shmOwner
    clink.shm.requests <> accelerator.io.shared_mem_req
    clink.shm.move <> accelerator.io.mvover
    accelerator.io.meshLocalBank <> clink.shm.local
    clink.shm.config <> accelerator.io.shared_config
    clink.shm.queryValid                        := accelerator.io.shared_query_valid
    clink.shm.queryVbank                        := accelerator.io.shared_query_vbank_id
    accelerator.io.shared_query_group_count     := clink.shm.queryGroups
    clink.shm.barrierArrive                     := accelerator.io.barrier_arrive
    accelerator.io.barrier_release              := clink.shm.barrierRelease
    accelerator.io.shared_bank_hashes.foreach(_ := clink.sharedHashes.get)
  }

}
