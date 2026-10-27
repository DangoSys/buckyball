// Copied from XiangShan src/main/scala/xiangshan/backend/fu/wrapper/CSR.scala and modified for Buckyball.
// XiangShan is licensed under Mulan PSL v2 (http://license.coscl.org.cn/MulanPSL2).

package examples.core.xiangshan.native

import chisel3._
import freechips.rocketchip.rocket.CSRs
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import utility._
import xiangshan._
import xiangshan.backend.fu.NewCSR._
import xiangshan.backend.fu.util._
import xiangshan.backend.fu.{FuConfig, FuncUnit}
import device._
import system.HasSoCParameter
import xiangshan.ExceptionNO._
import xiangshan.backend.Bundles._
import xiangshan.backend.decode.Imm_Z
import xiangshan.backend.fu.NewCSR.CSRBundles.PrivState
import xiangshan.backend.fu.NewCSR.CSRDefines.PrivMode
import xiangshan.backend.rob.RobPtr
import xiangshan.frontend.FtqPtr
import CSRConst._
import xiangshan.backend.fu.wrapper._
import examples.core.xiangshan.native.Command

class CSR(cfg: FuConfig)(implicit p: Parameters)
    extends FuncUnit(cfg)
    with HasCircularQueuePtrHelper
    with HasCriticalErrors
    with HasSoCParameter {
  val csrIn       = io.csrio.get
  val csrOut      = io.csrio.get
  val csrToDecode = io.csrToDecode.get

  val setFsDirty = csrIn.fpu.dirty_fs
  val setFflags  = csrIn.fpu.fflags

  val setVsDirty = csrIn.vpu.dirty_vs
  val setVstart  = csrIn.vpu.set_vstart
  val setVtype   = csrIn.vpu.set_vtype
  val setVxsat   = csrIn.vpu.set_vxsat
  val vlFromPreg = csrIn.vpu.vl

  val flushPipe = Wire(Bool())
  val flush     = io.flush.valid

  val timeAllowed = WireDefault(true.B)
  val isBall      = io.in.bits.ctrl.fuOpType === Command.opType

  /** Alias of input signals */
  val (valid, src1, imm, func) = (
    io.in.valid && timeAllowed && !isBall,
    io.in.bits.data.src(0),
    io.in.bits.data.imm(Imm_Z().len - 1, 0),
    io.in.bits.ctrl.fuOpType
  )

  // split imm/src1/rd from IMM_Z: src1/rd for tval
  val addr = Imm_Z().getCSRAddr(imm)
  val rd   = Imm_Z().getRD(imm)
  val rs1  = Imm_Z().getRS1(imm)
  val imm5 = Imm_Z().getImm5(imm)
  val csri = ZeroExt(imm5, XLEN)

  private val isEcall  = CSROpType.isSystemOp(func) && addr === privEcall
  private val isEbreak = CSROpType.isSystemOp(func) && addr === privEbreak
  private val isMNret  = CSROpType.isSystemOp(func) && addr === privMNret
  private val isMret   = CSROpType.isSystemOp(func) && addr === privMret
  private val isSret   = CSROpType.isSystemOp(func) && addr === privSret
  private val isDret   = CSROpType.isSystemOp(func) && addr === privDret
  private val isWfi    = CSROpType.isWfi(func)
  private val isCSRAcc = CSROpType.isCsrAccess(func)

  val csrMod       = Module(new NewCSR)
  val trapInstMod  = Module(new TrapInstMod)
  val trapTvalMod  = Module(new TrapTvalMod)
  val satpFlushMod = Module(new SatpFlushMod)

  private val privState = csrMod.io.status.privState
  // The real reg value in CSR, with no read mask
  private val regOut    = csrMod.io.out.bits.regOut
  private val src       = Mux(CSROpType.needImm(func), csri, src1)

  private val wdata = LookupTree(
    func,
    Seq(
      CSROpType.wrt  -> src1,
      CSROpType.set  -> (regOut | src1),
      CSROpType.clr  -> (regOut & (~src1).asUInt),
      CSROpType.wrti -> csri,
      CSROpType.seti -> (regOut | csri),
      CSROpType.clri -> (regOut & (~csri).asUInt)
    )
  )

  private val csrAccess = valid && CSROpType.isCsrAccess(func)

  private val csrWen = valid && (
    CSROpType.isCSRRW(func) ||
      CSROpType.isCSRRSorRC(func) && rs1 =/= 0.U
  )

  private val csrRen = valid && (
    CSROpType.isCSRRW(func) && rd =/= 0.U ||
      CSROpType.isCSRRSorRC(func)
  )

  private val waddrReg = RegEnable(addr, 0.U(12.W), io.in.fire)
  private val wdataReg = RegEnable(wdata, 0.U(64.W), io.in.fire)

  private val robIdxReg     = RegEnable(io.in.bits.ctrl.robIdx, io.in.fire)
  private val thisRobIdx    = Wire(new RobPtr)
  when(io.in.valid) {
    thisRobIdx := io.in.bits.ctrl.robIdx
  }.otherwise {
    thisRobIdx := robIdxReg
  }
  private val redirectFlush = thisRobIdx.needFlush(io.flush)

  val readsTime   = CSROpType.isCsrAccess(func) && addr === CSRs.time.U
  val timePending = RegInit(false.B)
  val timeFresh   = RegInit(false.B)
  val timeKilled  = RegInit(false.B)
  val timeOwner   = Reg(new RobPtr)
  timeAllowed                                                   := !timePending && (!readsTime || timeFresh)
  csrOut.timeRequest.valid                                      := io.in.valid && readsTime && !timePending && !timeFresh && !redirectFlush
  csrOut.timeRequest.bits                                       := true.B
  when(csrOut.timeRequest.fire) {
    timePending := true.B
    timeKilled  := false.B
    timeOwner   := io.in.bits.ctrl.robIdx
  }
  when(timePending && timeOwner.needFlush(io.flush))(timeKilled := true.B)
  when(io.csrin.get.clintTime.valid && timePending) {
    timePending := false.B
    timeFresh   := !timeKilled && !timeOwner.needFlush(io.flush)
  }
  when(io.in.fire && readsTime || redirectFlush)(timeFresh      := false.B)

  csrMod.io.in match {
    case in =>
      in.valid              := valid
      in.bits.wen           := csrWen
      in.bits.ren           := csrRen
      in.bits.op            := CSROpType.getCSROp(func)
      in.bits.addr          := addr
      in.bits.src           := src
      in.bits.wdata         := wdataReg
      in.bits.mret          := isMret
      in.bits.mnret         := isMNret
      in.bits.sret          := isSret
      in.bits.dret          := isDret
      in.bits.redirectFlush := redirectFlush
  }
  csrMod.io.trapInst                      := trapInstMod.io.currentTrapInst
  csrMod.io.fetchMalTval                  := trapTvalMod.io.tval
  csrMod.io.oldPrivSate                   := satpFlushMod.out.oldPrivState
  csrMod.io.oldSatpMode                   := satpFlushMod.out.oldSatpMode
  csrMod.io.oldVsatpMode                  := satpFlushMod.out.oldVsatpMode
  csrMod.io.fromMem.excpVA                := csrIn.memExceptionVAddr
  csrMod.io.fromMem.excpGPA               := csrIn.memExceptionGPAddr
  csrMod.io.fromMem.excpIsForVSnonLeafPTE := csrIn.memExceptionIsForVSnonLeafPTE

  csrMod.io.fromRob.trap.valid                         := csrIn.exception.valid
  csrMod.io.fromRob.trap.bits.pc                       := csrIn.exception.bits.pc
  csrMod.io.fromRob.trap.bits.instr                    := csrIn.exception.bits.instr
  csrMod.io.fromRob.trap.bits.pcGPA                    := csrIn.exception.bits.gpaddr
  // Todo: shrink the width of trap vector.
  // We use 64bits trap vector in CSR, and 24 bits exceptionVec in exception bundle.
  csrMod.io.fromRob.trap.bits.trapVec                  := csrIn.exception.bits.exceptionVec.asUInt
  csrMod.io.fromRob.trap.bits.isFetchBkpt              := csrIn.exception.bits.isPcBkpt
  csrMod.io.fromRob.trap.bits.singleStep               := csrIn.exception.bits.singleStep
  csrMod.io.fromRob.trap.bits.crossPageIPFFix          := csrIn.exception.bits.crossPageIPFFix
  csrMod.io.fromRob.trap.bits.isInterrupt              := csrIn.exception.bits.isInterrupt
  csrMod.io.fromRob.trap.bits.trigger                  := csrIn.exception.bits.trigger
  csrMod.io.fromRob.trap.bits.isHls                    := csrIn.exception.bits.isHls
  csrMod.io.fromRob.trap.bits.isFetchMalAddr           := csrIn.exception.bits.isFetchMalAddr
  csrMod.io.fromRob.trap.bits.satpFlushFirstFetchFault := csrIn.exception.bits.satpFlushFirstFetchFault
  csrMod.io.fromRob.trap.bits.isForVSnonLeafPTE        := csrIn.exception.bits.isForVSnonLeafPTE

  csrMod.io.fromRob.commit.fflags           := setFflags
  csrMod.io.fromRob.commit.fsDirty          := setFsDirty
  csrMod.io.fromRob.commit.vxsat.valid      := setVxsat.valid
  csrMod.io.fromRob.commit.vxsat.bits       := setVxsat.bits
  csrMod.io.fromRob.commit.vsDirty          := setVsDirty
  csrMod.io.fromRob.commit.vstart           := setVstart
  csrMod.io.fromRob.commit.vl               := vlFromPreg
  // Todo: correct vtype
  csrMod.io.fromRob.commit.vtype.valid      := setVtype.valid
  csrMod.io.fromRob.commit.vtype.bits.VILL  := setVtype.bits(XLEN - 1)
  csrMod.io.fromRob.commit.vtype.bits.VMA   := setVtype.bits(7)
  csrMod.io.fromRob.commit.vtype.bits.VTA   := setVtype.bits(6)
  csrMod.io.fromRob.commit.vtype.bits.VSEW  := setVtype.bits(5, 3)
  csrMod.io.fromRob.commit.vtype.bits.VLMUL := setVtype.bits(2, 0)

  csrMod.io.fromRob.commit.instNum.valid := true.B // Todo: valid control signal
  csrMod.io.fromRob.commit.instNum.bits  := csrIn.perf.retiredInstr

  csrMod.io.fromRob.robDeqPtr := csrIn.robDeqPtr

  csrMod.io.fromVecExcpMod.busy := io.csrin.get.fromVecExcpMod.busy

  csrMod.io.perf := csrIn.perf

  csrMod.platformIRP.MEIP      := csrIn.externalInterrupt.meip
  csrMod.platformIRP.MTIP      := csrIn.externalInterrupt.mtip
  csrMod.platformIRP.MSIP      := csrIn.externalInterrupt.msip
  csrMod.platformIRP.SEIP      := csrIn.externalInterrupt.seip
  csrMod.platformIRP.STIP      := false.B
  csrMod.platformIRP.VSEIP     := false.B // Todo
  csrMod.platformIRP.VSTIP     := false.B // Todo
  csrMod.platformIRP.debugIP   := csrIn.externalInterrupt.debug
  csrMod.nonMaskableIRP.NMI_43 := csrIn.externalInterrupt.nmi.nmi_43
  csrMod.nonMaskableIRP.NMI_31 := csrIn.externalInterrupt.nmi.nmi_31

  csrMod.io.fromTop.hartId                := io.csrin.get.hartId
  csrMod.io.fromTop.clintTime             := io.csrin.get.clintTime
  csrMod.io.fromTop.l2FlushDone           := io.csrin.get.l2FlushDone
  csrMod.io.fromTop.reset_mtvec.foreach(_ := io.csrin.get.reset_mtvec.get)
  csrMod.io.fromTop.criticalErrorState    := io.csrin.get.criticalErrorState
  private val csrModOutValid = csrMod.io.out.valid
  private val csrModOut      = csrMod.io.out.bits

  trapInstMod.io.fromDecode.trapInstInfo            := RegNextWithEnable(io.csrin.get.trapInstInfo, hasInit = true)
  trapInstMod.io.fromRob.flush.valid                := io.flush.valid
  trapInstMod.io.fromRob.flush.bits.ftqPtr          := io.flush.bits.ftqIdx
  trapInstMod.io.fromRob.flush.bits.ftqOffset       := io.flush.bits.ftqOffset
  trapInstMod.io.fromRob.isInterrupt.valid          := csrIn.exception.valid
  trapInstMod.io.fromRob.isInterrupt.bits           := csrIn.exception.bits.isInterrupt
  trapInstMod.io.faultCsrUop.valid                  := csrMod.io.out.valid && (csrMod.io.out.bits.EX_II || csrMod.io.out.bits.EX_VI)
  trapInstMod.io.faultCsrUop.bits.fuOpType          := DataHoldBypass(io.in.bits.ctrl.fuOpType, io.in.fire)
  trapInstMod.io.faultCsrUop.bits.imm               := DataHoldBypass(io.in.bits.data.imm, io.in.fire)
  trapInstMod.io.faultCsrUop.bits.ftqInfo.ftqPtr    := DataHoldBypass(io.in.bits.ctrl.ftqIdx.get, io.in.fire)
  trapInstMod.io.faultCsrUop.bits.ftqInfo.ftqOffset := DataHoldBypass(io.in.bits.ctrl.ftqOffset.get, io.in.fire)
  // Clear trap instruction when instruction fault trap(EX_II, EX_VI) occurs.
  trapInstMod.io.readClear                          := (csrMod.io.fromRob.trap match {
    case t =>
      t.valid && !t.bits.isInterrupt && (t.bits.trapVec(EX_II) || t.bits.trapVec(EX_VI))
  })

  trapTvalMod.io.targetPc.valid          := csrMod.io.out.bits.targetPcUpdate
  trapTvalMod.io.targetPc.bits           := csrMod.io.out.bits.targetPc
  trapTvalMod.io.clear                   := csrIn.exception.valid && csrIn.exception.bits.isFetchMalAddr
  trapTvalMod.io.fromCtrlBlock.flush     := io.flush
  trapTvalMod.io.fromCtrlBlock.robDeqPtr := io.csrio.get.robDeqPtr

  satpFlushMod.in.satp.valid  := csrMod.io.status.satp.valid
  satpFlushMod.in.satp.bits   := csrMod.io.status.satp.bits
  satpFlushMod.in.vsatp.valid := csrMod.io.status.vsatp.valid
  satpFlushMod.in.vsatp.bits  := csrMod.io.status.vsatp.bits
  satpFlushMod.in.privState   := csrMod.io.status.privState

  val imsic = Module(new aia.IMSIC_WRAP(soc.IMSICParams))
  imsic.fromCSR.addr.valid      := csrMod.toAIA.addr.valid
  imsic.fromCSR.addr.bits.addr  := csrMod.toAIA.addr.bits.addr
  imsic.fromCSR.addr.bits.virt  := csrMod.toAIA.addr.bits.v.asUInt.asBool
  imsic.fromCSR.addr.bits.priv  := aia.PrivType(csrMod.toAIA.addr.bits.prvm.asUInt)
  imsic.fromCSR.vgein           := csrMod.toAIA.vgein
  imsic.fromCSR.wdata.valid     := csrMod.toAIA.wdata.valid
  imsic.fromCSR.wdata.bits.op   := aia.OpType(csrMod.toAIA.wdata.bits.op)
  imsic.fromCSR.wdata.bits.data := csrMod.toAIA.wdata.bits.data
  imsic.fromCSR.claims(0)       := csrMod.toAIA.mClaim
  imsic.fromCSR.claims(1)       := csrMod.toAIA.sClaim
  imsic.fromCSR.claims(2)       := csrMod.toAIA.vsClaim

  csrMod.fromAIA.rdata.valid        := imsic.toCSR.rdata.valid
  csrMod.fromAIA.rdata.bits.data    := imsic.toCSR.rdata.bits
  csrMod.fromAIA.rdata.bits.illegal := imsic.toCSR.illegal
  csrMod.fromAIA.meip               := imsic.toCSR.pendings(0)
  csrMod.fromAIA.seip               := imsic.toCSR.pendings(1)
  csrMod.fromAIA.vseip              := imsic.toCSR.pendings(soc.IMSICParams.intFilesNum - 1, 2)
  csrMod.fromAIA.mtopei             := imsic.toCSR.topeis(0)
  csrMod.fromAIA.stopei             := imsic.toCSR.topeis(1)
  csrMod.fromAIA.vstopei            := imsic.toCSR.topeis(2)

  imsic.msiio.vld_req := io.csrin.get.msiInfo.valid
  imsic.msiio.data    := io.csrin.get.msiInfo.bits
  io.csrio.get.msiAck := imsic.msiio.vld_ack

  imsic.teemsiio.foreach { teemsiio =>
    teemsiio.vld_req           := io.csrin.get.teemsiInfo.get.valid
    teemsiio.data              := io.csrin.get.teemsiInfo.get.bits
    io.csrio.get.teemsiAck.get := teemsiio.vld_ack
  }
  csrMod.fromAIA.notice_pending := false.B
  imsic.sec.foreach { imsicsec =>
    imsicsec.cmode                := csrMod.io.tlb.mbmc.CMODE.asBool
    csrMod.fromAIA.notice_pending := imsicsec.notice_pending
  }

  private val exceptionVec = WireInit(0.U.asTypeOf(ExceptionVec())) // Todo:

  exceptionVec(EX_BP)     := DataHoldBypass(isEbreak, false.B, io.in.fire)
  exceptionVec(EX_MCALL)  := DataHoldBypass(isEcall && privState.isModeM, false.B, io.in.fire)
  exceptionVec(EX_HSCALL) := DataHoldBypass(isEcall && privState.isModeHS, false.B, io.in.fire)
  exceptionVec(EX_VSCALL) := DataHoldBypass(isEcall && privState.isModeVS, false.B, io.in.fire)
  exceptionVec(EX_UCALL)  := DataHoldBypass(isEcall && privState.isModeHUorVU, false.B, io.in.fire)
  exceptionVec(EX_II)     := csrMod.io.out.bits.EX_II
  exceptionVec(EX_VI)     := csrMod.io.out.bits.EX_VI

  val isXRet = valid && func === CSROpType.jmp && !isEcall && !isEbreak

  flushPipe := csrMod.io.out.bits.flushPipe

  // tlb
  val tlb = Wire(new TlbCsrBundle)
  tlb.satp.changed  := csrMod.io.tlb.satpASIDChanged
  tlb.satp.mode     := csrMod.io.tlb.satp.MODE.asUInt
  tlb.satp.asid     := csrMod.io.tlb.satp.ASID.asUInt
  tlb.satp.ppn      := csrMod.io.tlb.satp.PPN.asUInt
  tlb.vsatp.changed := csrMod.io.tlb.vsatpASIDChanged
  tlb.vsatp.mode    := csrMod.io.tlb.vsatp.MODE.asUInt
  tlb.vsatp.asid    := csrMod.io.tlb.vsatp.ASID.asUInt
  tlb.vsatp.ppn     := csrMod.io.tlb.vsatp.PPN.asUInt
  tlb.hgatp.changed := csrMod.io.tlb.hgatpVMIDChanged
  tlb.hgatp.mode    := csrMod.io.tlb.hgatp.MODE.asUInt
  tlb.hgatp.vmid    := csrMod.io.tlb.hgatp.VMID.asUInt
  tlb.hgatp.ppn     := csrMod.io.tlb.hgatp.PPN.asUInt
  tlb.mbmc.KEYIDEN  := csrMod.io.tlb.mbmc.KEYIDEN.asUInt
  tlb.mbmc.BME      := csrMod.io.tlb.mbmc.BME.asUInt
  tlb.mbmc.CMODE    := csrMod.io.tlb.mbmc.CMODE.asUInt
  tlb.mbmc.BCLEAR   := csrMod.io.tlb.mbmc.BCLEAR.asUInt
  tlb.mbmc.BMA      := csrMod.io.tlb.mbmc.BMA.asUInt

  // expose several csr bits for tlb
  tlb.priv.mxr          := csrMod.io.tlb.mxr
  tlb.priv.sum          := csrMod.io.tlb.sum
  tlb.priv.vmxr         := csrMod.io.tlb.vmxr
  tlb.priv.vsum         := csrMod.io.tlb.vsum
  tlb.priv.spvp         := csrMod.io.tlb.spvp
  tlb.priv.virt         := csrMod.io.tlb.dvirt
  tlb.priv.virt_changed := DataChanged(tlb.priv.virt)
  tlb.priv.imode        := csrMod.io.tlb.imode
  tlb.priv.dmode        := csrMod.io.tlb.dmode
  tlb.priv.debug        := csrMod.io.tlb.debug

  // Svpbmt extension enable
  tlb.mPBMTE := csrMod.io.tlb.mPBMTE
  tlb.hPBMTE := csrMod.io.tlb.hPBMTE

  // pointer masking extension
  tlb.pmm := csrMod.io.tlb.pmm

  /** Since some CSR read instructions are allowed to be pipelined, ready/valid signals should be modified */
  io.in.ready                        := csrMod.io.in.ready && timeAllowed // Includes asynchronous time reads.
  io.out.valid                       := csrModOutValid
  io.out.bits.ctrl.exceptionVec.get  := exceptionVec
  io.out.bits.ctrl.flushPipe.get     := flushPipe
  io.out.bits.ctrl.satpFlushPipe.get := csrMod.io.status.satp.valid || csrMod.io.status.vsatp.valid
  io.out.bits.res.data               := csrMod.io.out.bits.rData

  /** initialize NewCSR's io_out_ready from wrapper's io */
  csrMod.io.out.ready := io.out.ready

  io.out.bits.res.redirect.get.valid := io.out.valid && RegEnable(isXRet, false.B, io.in.fire)
  val redirect = io.out.bits.res.redirect.get.bits
  redirect                       := 0.U.asTypeOf(redirect)
  redirect.level                 := RedirectLevel.flushAfter
  redirect.robIdx                := robIdxReg
  redirect.ftqIdx                := RegEnable(io.in.bits.ctrl.ftqIdx.get, io.in.fire)
  redirect.ftqOffset             := RegEnable(io.in.bits.ctrl.ftqOffset.get, io.in.fire)
  redirect.cfiUpdate.predTaken   := true.B
  redirect.cfiUpdate.taken       := true.B
  redirect.cfiUpdate.target      := csrMod.io.out.bits.targetPc.pc
  redirect.cfiUpdate.backendIPF  := csrMod.io.out.bits.targetPc.raiseIPF
  redirect.cfiUpdate.backendIAF  := csrMod.io.out.bits.targetPc.raiseIAF
  redirect.cfiUpdate.backendIGPF := csrMod.io.out.bits.targetPc.raiseIGPF
  // Only mispred will send redirect to frontend
  redirect.cfiUpdate.isMisPred   := false.B

  connectNonPipedCtrlSingal

  val commandIdle :: commandDrain :: commandSend :: commandWait :: commandDone :: Nil = Enum(5)
  val commandState                                                                    = RegInit(commandIdle)
  val command                                                                         = Reg(chiselTypeOf(csrOut.commands.request.bits))
  val commandError                                                                    = RegInit(false.B)
  val commandOwner                                                                    = Reg(new RobPtr)
  csrOut.commandFlushStores                                                     := commandState === commandDrain
  csrOut.commands.request.valid                                                 := commandState === commandSend
  csrOut.commands.request.bits                                                  := command
  when(isBall)(io.in.ready                                                      := commandState === commandIdle && !timePending && !timeFresh)
  when(io.in.fire && isBall) {
    assert(!csrMod.io.out.valid)
    command.instruction := Cat(Imm_Z().getCSRAddr(imm), Imm_Z().getRS1(imm), 3.U(3.W), 0.U(5.W), "h7b".U(7.W))
    command.pc          := io.in.bits.data.pc.get
    command.rs1         := io.in.bits.data.src(0)
    command.rs2         := io.in.bits.data.src(1)
    command.context     := csrMod.io.accelerator
    commandOwner        := io.in.bits.ctrl.robIdx
    commandError        := csrMod.io.accelerator.virt
    commandState        := Mux(csrMod.io.accelerator.virt, commandDone, commandDrain)
  }
  when(commandState =/= commandIdle)(io.in.ready                                := false.B)
  when(commandState === commandDrain && csrOut.commandStoresEmpty)(commandState := commandSend)
  when(csrOut.commands.request.fire)(commandState                               := commandWait)
  when(commandState === commandWait && csrOut.commands.complete.valid) {
    commandError := csrOut.commands.complete.bits
    commandState := commandDone
  }
  when(commandState === commandDone) {
    io.out.valid                                    := true.B
    io.out.bits.res.data                            := 0.U
    io.out.bits.res.redirect.get.valid              := false.B
    io.out.bits.ctrl.exceptionVec.get               := 0.U.asTypeOf(io.out.bits.ctrl.exceptionVec.get)
    // Failed accelerator operations trap instead of retiring as successful commands.
    io.out.bits.ctrl.exceptionVec.get(illegalInstr) := commandError
    io.out.bits.ctrl.flushPipe.get                  := false.B
    io.out.bits.ctrl.satpFlushPipe.get              := false.B
    when(io.out.fire)(commandState                  := commandIdle)
  }
  when(commandState =/= commandIdle)(assert(!commandOwner.needFlush(io.flush)))

  override val criticalErrors = csrMod.getCriticalErrors
  generateCriticalErrors()

  // Todo: summerize all difftest skip condition
  csrOut.isPerfCnt  := io.out.valid && csrMod.io.out.bits.isPerfCnt && RegEnable(
    func =/= CSROpType.jmp,
    false.B,
    io.in.fire
  )
  csrOut.fpu.frm    := csrMod.io.status.fpState.frm.asUInt
  csrOut.vpu.vstart := csrMod.io.status.vecState.vstart.asUInt
  csrOut.vpu.vxrm   := csrMod.io.status.vecState.vxrm.asUInt

  csrOut.isXRet := isXRet

  csrOut.trapTarget := csrMod.io.out.bits.targetPc
  csrOut.interrupt  := csrMod.io.status.interrupt
  csrOut.wfi_event  := csrMod.io.status.wfiEvent

  csrOut.tlb := tlb

  csrOut.debugMode := csrMod.io.status.debugMode

  csrOut.traceCSR := csrMod.io.status.traceCSR

  csrOut.customCtrl match {
    case custom =>
      custom.pf_ctrl                          := csrMod.io.status.custom.pf_ctrl
      // Load violation predictor
      custom.lvpred_disable                   := csrMod.io.status.custom.lvpred_disable
      custom.no_spec_load                     := csrMod.io.status.custom.no_spec_load
      custom.storeset_wait_store              := csrMod.io.status.custom.storeset_wait_store
      custom.storeset_no_fast_wakeup          := csrMod.io.status.custom.storeset_no_fast_wakeup
      custom.lvpred_timeout                   := csrMod.io.status.custom.lvpred_timeout
      // Branch predictor
      custom.bp_ctrl                          := csrMod.io.status.custom.bp_ctrl
      // Memory Block
      custom.sbuffer_timeout                  := csrMod.io.status.custom.sbuffer_timeout
      custom.sbuffer_threshold                := csrMod.io.status.custom.sbuffer_threshold
      custom.ldld_vio_check_enable            := csrMod.io.status.custom.ldld_vio_check_enable
      custom.soft_prefetch_enable             := csrMod.io.status.custom.soft_prefetch_enable
      custom.cache_error_enable               := csrMod.io.status.custom.cache_error_enable
      custom.uncache_write_outstanding_enable := csrMod.io.status.custom.uncache_write_outstanding_enable
      custom.hd_misalign_st_enable            := csrMod.io.status.custom.hd_misalign_st_enable
      custom.hd_misalign_ld_enable            := csrMod.io.status.custom.hd_misalign_ld_enable
      custom.power_down_enable                := csrMod.io.status.custom.power_down_enable
      custom.commit_stuck_check_enable        := csrMod.io.status.custom.commit_stuck_check_enable
      custom.flush_l2_enable                  := csrMod.io.status.custom.flush_l2_enable
      // Rename
      custom.fusion_enable                    := csrMod.io.status.custom.fusion_enable
      custom.wfi_enable                       := csrMod.io.status.custom.wfi_enable
      // distribute csr write signal
      // write to frontend and memory
      custom.distribute_csr.w.valid           := csrMod.io.distributedWenLegal
      custom.distribute_csr.w.bits.addr       := waddrReg
      custom.distribute_csr.w.bits.data       := wdataReg
      // rename single step
      custom.singlestep                       := csrMod.io.status.singleStepFlag
      // trigger
      custom.frontend_trigger                 := csrMod.io.status.frontendTrigger
      custom.mem_trigger                      := csrMod.io.status.memTrigger
      // virtual mode
      custom.virtMode                         := csrMod.io.status.privState.V.asBool
      // xstatus.fs field is off
      custom.fsIsOff                          := csrMod.io.toDecode.illegalInst.fsIsOff
  }

  csrOut.instrAddrTransType := csrMod.io.status.instrAddrTransType
  csrOut.criticalErrorState := csrMod.io.status.criticalErrorState

  csrToDecode := csrMod.io.toDecode
}
