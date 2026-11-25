package framework.hart.configs

import framework.generic.contract.configs.ThreadSlot

import framework.generic.acc.configs.AccParam
import buckyball.config.{Design, HartPlacement}
import framework.tlp.tbus.router.configs.{Params => RouterParam}
import scala.jdk.CollectionConverters._
import framework.generic.memory.spm

import framework.hart.execution.instruction.configs.{Params => InstructionParams}
import framework.hart.execution.integer.configs.{Params => IntegerParams}
import framework.hart.execution.muldiv.configs.{Params => MulDivParams}
import framework.hart.execution.fpu.configs.{Params => FpuParams}
import framework.hart.execution.privileged.configs.{Params => PrivilegedParams}
import framework.hart.execution.register_file.configs.{Params => RegisterFileParams}
import framework.tlp.controller.admission.configs.{Params => AdmissionParams}
import framework.hart.hlink.configs.{Params => HLinkParams}
import framework.tlp.tbus.configs.{Params => TBusParams}
import framework.tlp.tss.configs.{Params => TaskStorageParams}
import framework.tlp.shm.configs.Geometry
import framework.generic.bus.axi4.{InterconnectParam, Params => AxiParam}
import framework.hart.execution.configs.ExecutionParam
import framework.hart.memory.configs.MemoryParam
import framework.tlp.controller.configs.{CommandsParam, ControllerParam, TasksParam}
import framework.generic.contract.memory.access.preflight.configs.PmpParams

case class HartParam(
  hartId:       Int,
  resetVector:  BigInt,
  mtimeAddress: BigInt,
  hlink:        HLinkParams,
  tbus:         Option[TBusParams],
  tss:          Option[TaskStorageParams],
  shm:          Option[Geometry],
  interconnect: InterconnectParam,
  threads:      Seq[ThreadSlot],
  acc:          Option[AccParam],
  memory:       MemoryParam,
  instruction:  InstructionParams,
  integer:      IntegerParams,
  muldiv:       MulDivParams,
  fpu:          FpuParams,
  privileged:   PrivilegedParams,
  registerFile: RegisterFileParams,
  admission:    AdmissionParams) {
  require(tss.forall(_.ports == threads.size))
  require(threads.isEmpty || interconnect.masters == threads.size)
  require(tbus.forall(bus => threads.isEmpty || (bus.participants == threads.map(_.id) && bus.slots == threads.size)))

  def controller: ControllerParam = {
    val taskParams = Option.when(threads.nonEmpty)(
      TasksParam(threads.map(_.link), threads.map(_.signature), tss.get, privileged.permission)
    )
    ControllerParam(
      CommandsParam(
        memory.dcache.tracking,
        integer.xLen,
        privileged.permission,
        admission.moves,
        threads.nonEmpty,
        hlink.service
      ),
      taskParams
    )
  }

  def execution(directAccelerator: Boolean): ExecutionParam = ExecutionParam(
    hartId,
    resetVector,
    threads.nonEmpty || hlink.service || directAccelerator,
    directAccelerator,
    admission.moves,
    memory.dcache.tracking,
    mtimeAddress,
    memory.memoryAccess,
    instruction,
    registerFile,
    integer,
    muldiv,
    fpu,
    privileged
  )

}

object HartParam {

  def fromProto(
    design:       Design,
    hart:         HartPlacement,
    externalHome: Boolean
  ): HartParam = {
    val threads = hart.getThreadIndicesList.asScala.map { index =>
      val thread = design.getThreads(index.intValue)
      require(thread.getIndex == index.intValue)
      ThreadSlot.fromProto(thread, hart)
    }.toSeq
    val hartp   = fromProto(hart, threads)
    if (externalHome) require(hart.hasNodeId && hart.getNodeId > 0, "Hart coherence node ID is required")
    // The hart's port to the DDR carries the DDR's beat: `beatRows` bank rows.
    val ddr     = hartp.hlink.ddr
    hartp.copy(hlink =
      hartp.hlink.copy(
        ddr = ddr.copy(dataBits = ddr.dataBits * design.getDevice.getDdr.getBeatRows),
        backing = !externalHome,
        // The threads' DMA and the service payload's go to the DDR beside the caches.
        pages = externalHome && (threads.nonEmpty || hartp.hlink.service),
        requesterIds =
          if (externalHome)
            Seq(hart.getNodeId)
          else Seq.empty
      )
    )
  }

  def fromProto(p: HartPlacement, threads: Seq[ThreadSlot]): HartParam = {
    require(
      p.hasMemory && p.hasParam && p.hasSharedMem && (threads.isEmpty || p.hasTss),
      "Hart configuration is incomplete"
    )
    require(p.getThreadIndicesCount == threads.size, "Hart thread configuration is incomplete")
    val accp         = Option.when(p.hasAccelerator)(AccParam.fromProto(p))
    val memp         = MemoryParam.fromProto(p.getMemory, threads.size)
    val cp           = p.getParam
    val permission   = PmpParams(cp.getPaddrBits, cp.getNPmps)
    val shared       = p.getSharedMem
    val tss          = p.getTss
    val ids          = threads.map(_.id)
    val directShared = threads.isEmpty && accp.exists(_.memDomain.sharedEnable)
    val participants = if (directShared) Seq(0) else ids
    val crossbar     = AxiParam.fromProto(p.getInterconnect.getAxi)
    require(ids == ids.indices, "TBus thread IDs must be ordered and contiguous")
    HartParam(
      hartId = p.getHartId,
      resetVector = BigInt(java.lang.Long.toUnsignedString(p.getResetVector)),
      mtimeAddress = BigInt(java.lang.Long.toUnsignedString(p.getMtimeAddress)),
      hlink = HLinkParams(
        1,
        true,
        Seq.empty,
        memp.memoryAccess.memory.chi,
        memp.uncached.axi,
        accp.map(_.transport.axi).getOrElse(crossbar),
        p.getService,
        false
      ),
      tbus = Option.when(p.hasTbus)(TBusParams(
        RouterParam(0),
        participants,
        participants.size,
        p.getTbus.getBankNodesList.asScala.map(_.intValue).toSeq,
        !directShared && p.getController.getMoves
      )),
      tss = Option.when(threads.nonEmpty)(TaskStorageParams(
        spm.Params(BigInt(java.lang.Long.toUnsignedString(tss.getBase)), tss.getBytes, tss.getDataBits),
        ids.size
      )),
      shm = Option.when(shared.getEnable)(Geometry(
        shared.getBankWidth,
        shared.getBankEntries,
        shared.getEntries / shared.getBankEntries
      )),
      interconnect = InterconnectParam(crossbar, ids.size, Some(p.getInterconnect.getSlots)),
      threads = threads,
      acc = accp,
      memory = memp,
      instruction = InstructionParams(cp.getXLen),
      integer = IntegerParams(cp.getXLen),
      muldiv = MulDivParams(cp.getXLen),
      fpu = FpuParams(cp.getXLen, p.getFpu.getFLen),
      privileged = PrivilegedParams(permission),
      registerFile = RegisterFileParams(cp.getXLen),
      admission = AdmissionParams(permission, p.getController.getMoves && threads.isEmpty)
    )
  }

}
