package framework.core.configs

import buckyball.config.{CorePlacement, Design}
import framework.acc.configs.AccParam
import framework.core.tbus.router.configs.{Params => RouterParam}
import scala.jdk.CollectionConverters._
import framework.ip.memory.spm
import framework.core.memory.access.preflight.configs.PmpParams

import framework.core.execution.fetch.configs.{Params => FetchParams}
import framework.core.execution.instruction.configs.{Params => InstructionParams}
import framework.core.execution.integer.configs.{Params => IntegerParams}
import framework.core.execution.muldiv.configs.{Params => MulDivParams}
import framework.core.execution.fpu.configs.{Params => FpuParams}
import framework.core.execution.privileged.configs.{Params => PrivilegedParams}
import framework.core.execution.register_file.configs.{Params => RegisterFileParams}
import framework.core.controller.admission.configs.{Params => AdmissionParams}
import framework.thread.configs.ThreadParam
import framework.acc.frontend.configs.FrontendParam
import framework.acc.memdomain.configs.MemDomainParam
import framework.core.clink.configs.{Params => CLinkParams}
import framework.core.tbus.configs.{Params => TBusParams}
import framework.core.tss.configs.{Params => TaskStorageParams}
import framework.device.cbus.C2CParams
import framework.ip.bus.chi.rnf.RnfParams
import framework.ip.bus.axi4.{InterconnectParam, Params => AxiParam}
import framework.core.execution.configs.ExecutionParam
import framework.core.memory.configs.MemoryParam
import framework.core.controller.configs.{CommandsParam, ControllerParam, TasksParam}

case class CoreParam(
  hartId:              Int,
  resetVector:         BigInt,
  mtimeAddress:        BigInt,
  clink:               CLinkParams,
  tbus:                Option[TBusParams],
  tss:                 Option[TaskStorageParams],
  c2c:                 C2CParams,
  transferLocalAccess: Boolean,
  transferLeaseAccess: Boolean,
  dmaCache:            Option[RnfParams],
  interconnect:        InterconnectParam,
  threads:             Seq[ThreadParam],
  acc:                 Option[AccParam],
  memory:              MemoryParam,
  fetch:               FetchParams,
  instruction:         InstructionParams,
  integer:             IntegerParams,
  muldiv:              MulDivParams,
  fpu:                 FpuParams,
  privileged:          PrivilegedParams,
  registerFile:        RegisterFileParams,
  admission:           AdmissionParams) {

  // TBus uses the bank/tag widths, not a thread's private scheduling policy.
  def sharedFrontend: FrontendParam = threads.map(_.acc.get.frontend).reduce { (left, right) =>
    left.copy(
      bank_id_len = math.max(left.bank_id_len, right.bank_id_len),
      rob_entries = math.max(left.rob_entries, right.rob_entries)
    )
  }

  // Physical shared SRAM geometry must agree. Private capacity and virtual namespaces may differ.
  def sharedMemory: MemDomainParam = threads.map(_.acc.get.memDomain).reduce { (left, right) =>
    require(
      (
        left.bankWidth,
        left.bankMaskLen,
        left.sharedEnable,
        left.sharedBankNum,
        left.sharedBankEntries,
        left.sharedInputChannels
      ) ==
        (right.bankWidth, right.bankMaskLen, right.sharedEnable, right.sharedBankNum, right.sharedBankEntries, right.sharedInputChannels),
      "Threads attached to one TBus must agree on the shared SRAM interface"
    )
    left.copy(
      bankNum = math.max(left.bankNum, right.bankNum),
      bankEntries = left.sharedBankEntries,
      virtualBankCount = math.max(left.virtualBankCount, right.virtualBankCount)
    )
  }

  def controller: ControllerParam = {
    val taskParams = Option.when(threads.nonEmpty)(
      TasksParam(threads.map(_.ant.get), threads.map(_.signature), tss.get, privileged.permission)
    )
    ControllerParam(
      CommandsParam(
        memory.dcache.tracking,
        integer.xLen,
        privileged.permission,
        admission.moves,
        threads.nonEmpty,
        clink.shared
      ),
      taskParams
    )
  }

  def execution(directAccelerator: Boolean): ExecutionParam = ExecutionParam(
    hartId,
    resetVector,
    threads.nonEmpty || clink.shared || directAccelerator,
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

object CoreParam {

  def fromProto(
    design:       Design,
    core:         CorePlacement,
    externalHome: Boolean
  ): CoreParam = {
    val threads = core.getThreadIndicesList.asScala.map { index =>
      val thread = design.getThreads(index.intValue)
      require(thread.getIndex == index.intValue)
      ThreadParam.fromProto(thread, core)
    }.toSeq
    val corep   = fromProto(core, threads)
    corep.copy(clink =
      corep.clink.copy(
        backing = !externalHome,
        requesterIds =
          if (externalHome)
            Seq(corep.memory.icache.nodeId, corep.memory.dcache.cache.nodeId) ++ corep.dmaCache.map(_.nodeId)
          else Seq.empty
      )
    )
  }

  def fromProto(p: CorePlacement, threads: Seq[ThreadParam]): CoreParam = {
    require(
      p.hasMemory && p.hasParam && p.hasSharedMem && (threads.isEmpty || p.hasTss),
      "Core configuration is incomplete"
    )
    require(p.getThreadIndicesCount == threads.size, "Core thread configuration is incomplete")
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
    val c2c          = p.getC2C
    val c2cAxi       = AxiParam.fromProto(c2c.getAxi)
    require(ids == ids.indices, "TBus thread IDs must be ordered and contiguous")
    CoreParam(
      hartId = p.getHartId,
      resetVector = BigInt(java.lang.Long.toUnsignedString(p.getResetVector)),
      mtimeAddress = BigInt(java.lang.Long.toUnsignedString(p.getMtimeAddress)),
      clink = CLinkParams(
        1,
        true,
        Seq.empty,
        memp.memoryAccess.memory.chi,
        memp.uncached.axi,
        accp.map(_.transport.axi).getOrElse(crossbar),
        Option.when(c2c.getEnabled)(c2cAxi)
      ),
      tbus = Option.when(p.hasTbus)(TBusParams(
        RouterParam(if (c2c.getEnabled) 3 + (if (directShared) 1 else 0) else 0),
        participants,
        participants.size,
        p.getTbus.getBankNodesList.asScala.map(_.intValue).toSeq,
        !directShared && p.getController.getMoves
      )),
      tss = Option.when(threads.nonEmpty)(TaskStorageParams(
        spm.Params(BigInt(java.lang.Long.toUnsignedString(tss.getBase)), tss.getBytes, tss.getDataBits),
        ids.size
      )),
      c2c = C2CParams(
        BigInt(shared.getEntries) * (shared.getBankWidth / 8),
        shared.getBankEntries * (shared.getBankWidth / 8),
        c2c.getPeerCores,
        c2cAxi
      ),
      transferLocalAccess = directShared && c2c.getEnabled,
      transferLeaseAccess = c2c.getEnabled,
      dmaCache = Option.when(p.hasDmaCache)(RnfParams.fromProto(p.getDmaCache, memp.home.memory.chi)),
      interconnect = InterconnectParam(crossbar, ids.size, Some(p.getInterconnect.getSlots)),
      threads = threads,
      acc = accp,
      memory = memp,
      fetch = FetchParams(p.getFetch.getInstructionBytes),
      instruction = InstructionParams(cp.getXLen),
      integer = IntegerParams(cp.getXLen),
      muldiv = MulDivParams(cp.getXLen),
      fpu = FpuParams(cp.getXLen, p.getFpu.getFLen),
      privileged = PrivilegedParams(permission),
      registerFile = RegisterFileParams(cp.getXLen),
      admission = AdmissionParams(permission, p.getController.getMoves)
    )
  }

}
