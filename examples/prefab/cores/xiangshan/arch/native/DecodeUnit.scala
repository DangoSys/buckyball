// Copied from XiangShan src/main/scala/xiangshan/backend/decode/DecodeUnit.scala and modified for Buckyball.
// XiangShan is licensed under Mulan PSL v2 (http://license.coscl.org.cn/MulanPSL2).

package examples.core.xiangshan.native

import org.chipsalliance.cde.config.Parameters
import chisel3._
import chisel3.util._
import freechips.rocketchip.rocket.CSRs
import freechips.rocketchip.rocket.Instructions._
import freechips.rocketchip.rocket.CustomInstructions._
import freechips.rocketchip.util.uintToBitPat
import utility._
import utils._
import xiangshan.ExceptionNO.{breakPoint, illegalInstr, virtualInstr, EX_II}
import xiangshan._
import xiangshan.backend.fu.FuType
import xiangshan.backend.Bundles.{DecodedInst, DynInst, StaticInst}
import xiangshan.backend.decode.isa.PseudoInstructions
import xiangshan.backend.decode.isa.bitfield.{InstVType, OPCODE5Bit, XSInstBitFields}
import xiangshan.backend.fu.vector.Bundles.{VType, Vl}
import xiangshan.backend.fu.wrapper.CSRToDecode
import xiangshan.backend.decode.Zimop._
import yunsuan.{VfaluType, VfcvtType}
import xiangshan.backend.decode._
import examples.core.xiangshan.native.{Command}

/**
 * Decode unit that takes in a single CtrlFlow and generates a CfCtrl.
 */
class DecodeUnit(implicit p: Parameters) extends XSModule with DecodeUnitConstants {
  val io = IO(new DecodeUnitIO)

  val ctrl_flow = io.enq.ctrlFlow // input with RVC Expanded

  private val inst: XSInstBitFields = io.enq.ctrlFlow.instr.asTypeOf(new XSInstBitFields)

  // The accelerator command sits right before CSRRW in the integer table.
  val decode_table: Array[(BitPat, List[BitPat])] =
    XDecode.table.patch(XDecode.decodeArray.indexWhere(_._1 == CSRRW), Seq(Command.decode), 0) ++
      FpDecode.table ++
//    FDivSqrtDecode.table ++
      BitmanipDecode.table ++
      ScalarCryptoDecode.table ++
      XSDebugDecode.table ++
      CBODecode.table ++
      SvinvalDecode.table ++
      HypervisorDecode.table ++
      VecDecoder.table ++
      ZicondDecode.table ++
      ZimopDecode.table ++
      ZfaDecode.table

  require(decode_table.forall(_._1.width == 32), "Decoded instructions must be 32 bits after RVC expansion")
  require(decode_table.map(_._2.length == 15).reduce(_ && _), "Decode tables have different column size")
  // assertion for LUI: only LUI should be assigned `selImm === SelImm.IMM_U && fuType === FuType.alu`
  val luiMatch = (t: Seq[BitPat]) => t(3).value == FuType.alu.ohid && t.reverse.head.value == SelImm.IMM_U.litValue
  val luiTable = decode_table.filter(t => luiMatch(t._2)).map(_._1).distinct
  assert(luiTable.length == 1 && luiTable.head == LUI, "Conflicts: LUI is determined by FuType and SelImm in Dispatch")

  // output
  val decodedInst: DecodedInst = Wire(new DecodedInst()).decode(ctrl_flow.instr, decode_table)

  val fpDecoder = Module(new FPDecoder)
  fpDecoder.io.instr     := ctrl_flow.instr
  decodedInst.fpu        := fpDecoder.io.fpCtrl
  decodedInst.fpu.wflags := fpDecoder.io.fpCtrl.wflags || decodedInst.wfflags

  decodedInst.connectStaticInst(io.enq.ctrlFlow)

  decodedInst.uopIdx   := 0.U
  decodedInst.firstUop := true.B
  decodedInst.lastUop  := true.B
  decodedInst.numUops  := 1.U
  decodedInst.numWB    := 1.U

  val isZimop = (BitPat("b1?00??0111??_?????_100_?????_1110011") === ctrl_flow.instr) ||
    (BitPat("b1?00??1?????_?????_100_?????_1110011") === ctrl_flow.instr)

  val isMove = BitPat("b000000000000_?????_000_?????_0010011") === ctrl_flow.instr
  // temp decode zimop as move
  decodedInst.isMove := (isMove || isZimop) && ctrl_flow.instr(RD_MSB, RD_LSB) =/= 0.U && !io.csrCtrl.singlestep

  // fmadd - b1000011
  // fmsub - b1000111
  // fnmsub- b1001011
  // fnmadd- b1001111
  private val isFMA    = inst.OPCODE === BitPat("b100??11")
  private val isVppu   = FuType.isVppu(decodedInst.fuType)
  private val isVecOPF = FuType.isVecOPF(decodedInst.fuType)

  // read src1~3 location
  decodedInst.lsrc(0) := inst.RS1
  decodedInst.lsrc(1) := inst.RS2
  // src(2) of fma is fs3, src(2) of vector inst is old vd
  decodedInst.lsrc(2) := Mux(isFMA, inst.FS3, inst.VD)
  decodedInst.lsrc(3) := V0_IDX.U
  decodedInst.lsrc(4) := Vl_IDX.U

  // read dest location
  decodedInst.ldest := inst.RD

  // init v0Wen vlWen
  decodedInst.v0Wen := false.B
  decodedInst.vlWen := false.B

  val isCsr  = inst.OPCODE5Bit === OPCODE5Bit.SYSTEM && inst.FUNCT3(1, 0) =/= 0.U
  val isCsrr = isCsr && inst.FUNCT3 === BitPat("b?1?") && inst.RS1 === 0.U
  val isCsrw = isCsr && inst.FUNCT3 === BitPat("b?01") && inst.RD === 0.U
  dontTouch(isCsrr)
  dontTouch(isCsrw)

  // for csrr vl instruction, convert to vsetvl
  val isCsrrVlenb = isCsrr && inst.CSRIDX === CSRs.vlenb.U
  val isCsrrVl    = isCsrr && inst.CSRIDX === CSRs.vl.U

  private val isCboClean = CBO_CLEAN === io.enq.ctrlFlow.instr
  private val isCboFlush = CBO_FLUSH === io.enq.ctrlFlow.instr
  private val isCboInval = CBO_INVAL === io.enq.ctrlFlow.instr
  private val isCboZero  = CBO_ZERO === io.enq.ctrlFlow.instr

  // Note that rnum of aes64ks1i must be in the range 0x0..0xA. The values 0xB..0xF are reserved.
  private val isAes64ks1iIllegal =
    FuType.FuTypeOrR(
      decodedInst.fuType,
      FuType.bku
    ) && (decodedInst.fuOpType === BKUOpType.aes64ks1i) && inst.isRnumIllegal

  private val isAmocasQ        =
    FuType.FuTypeOrR(decodedInst.fuType, FuType.mou) && decodedInst.fuOpType === LSUOpType.amocas_q
  private val isAmocasQIllegal = isAmocasQ && (inst.RD(0) === 1.U || inst.RS2(0) === 1.U)

  private val exceptionII =
    decodedInst.selImm === SelImm.INVALID_INSTR ||
      io.fromCSR.illegalInst.sfenceVMA && FuType.FuTypeOrR(
        decodedInst.fuType,
        FuType.fence
      ) && decodedInst.fuOpType === FenceOpType.sfence ||
      io.fromCSR.illegalInst.sfencePart && FuType.FuTypeOrR(
        decodedInst.fuType,
        FuType.fence
      ) && decodedInst.fuOpType === FenceOpType.nofence ||
      io.fromCSR.illegalInst.hfenceGVMA && FuType.FuTypeOrR(
        decodedInst.fuType,
        FuType.fence
      ) && decodedInst.fuOpType === FenceOpType.hfence_g ||
      io.fromCSR.illegalInst.hfenceVVMA && FuType.FuTypeOrR(
        decodedInst.fuType,
        FuType.fence
      ) && decodedInst.fuOpType === FenceOpType.hfence_v ||
      io.fromCSR.illegalInst.hlsv && FuType.FuTypeOrR(decodedInst.fuType, FuType.ldu) && (LSUOpType.isHlv(
        decodedInst.fuOpType
      ) || LSUOpType.isHlvx(decodedInst.fuOpType)) ||
      io.fromCSR.illegalInst.hlsv && FuType.FuTypeOrR(decodedInst.fuType, FuType.stu) && LSUOpType.isHsv(
        decodedInst.fuOpType
      ) ||
      io.fromCSR.illegalInst.fsIsOff && (
        FuType.FuTypeOrR(decodedInst.fuType, FuType.fpOP ++ Seq(FuType.f2v)) ||
          (FuType.FuTypeOrR(
            decodedInst.fuType,
            FuType.ldu
          ) && (decodedInst.fuOpType === LSUOpType.lh || decodedInst.fuOpType === LSUOpType.lw || decodedInst.fuOpType === LSUOpType.ld) ||
            FuType.FuTypeOrR(
              decodedInst.fuType,
              FuType.stu
            ) && (decodedInst.fuOpType === LSUOpType.sh || decodedInst.fuOpType === LSUOpType.sw || decodedInst.fuOpType === LSUOpType.sd)) && decodedInst.instr(
            2
          ) ||
          inst.isOPFVF || inst.isOPFVV
      ) ||
      io.fromCSR.illegalInst.vsIsOff && (FuType.FuTypeOrR(
        decodedInst.fuType,
        FuType.vecAll
      ) || isCsrrVl || isCsrrVlenb) ||
      io.fromCSR.illegalInst.wfi && FuType.FuTypeOrR(decodedInst.fuType, FuType.csr) && CSROpType.isWfi(
        decodedInst.fuOpType
      ) ||
      io.fromCSR.illegalInst.wrs_nto && FuType.FuTypeOrR(decodedInst.fuType, FuType.csr) && CSROpType.isWrsNto(
        decodedInst.fuOpType
      ) ||
      (decodedInst.needFrm.scalaNeedFrm || FuType.isScalaNeedFrm(
        decodedInst.fuType
      )) && (((decodedInst.fpu.rm === 5.U) || (decodedInst.fpu.rm === 6.U)) || ((decodedInst.fpu.rm === 7.U) && io.fromCSR.illegalInst.frm)) ||
      (decodedInst.needFrm.vectorNeedFrm || FuType.isVectorNeedFrm(decodedInst.fuType)) && io.fromCSR.illegalInst.frm ||
      (io.fromCSR.illegalInst.cboZ || !HasCMO.B) && isCboZero ||
      (io.fromCSR.illegalInst.cboCF || !HasCMO.B) && (isCboClean || isCboFlush) ||
      (io.fromCSR.illegalInst.cboI || !HasCMO.B) && isCboInval ||
      isAes64ks1iIllegal ||
      isAmocasQIllegal

  private val exceptionVI =
    io.fromCSR.virtualInst.sfenceVMA && FuType.FuTypeOrR(
      decodedInst.fuType,
      FuType.fence
    ) && decodedInst.fuOpType === FenceOpType.sfence ||
      io.fromCSR.virtualInst.sfencePart && FuType.FuTypeOrR(
        decodedInst.fuType,
        FuType.fence
      ) && decodedInst.fuOpType === FenceOpType.nofence ||
      io.fromCSR.virtualInst.hfence && FuType.FuTypeOrR(
        decodedInst.fuType,
        FuType.fence
      ) && (decodedInst.fuOpType === FenceOpType.hfence_g || decodedInst.fuOpType === FenceOpType.hfence_v) ||
      io.fromCSR.virtualInst.hlsv && FuType.FuTypeOrR(decodedInst.fuType, FuType.ldu) && (LSUOpType.isHlv(
        decodedInst.fuOpType
      ) || LSUOpType.isHlvx(decodedInst.fuOpType)) ||
      io.fromCSR.virtualInst.hlsv && FuType.FuTypeOrR(decodedInst.fuType, FuType.stu) && LSUOpType.isHsv(
        decodedInst.fuOpType
      ) ||
      io.fromCSR.virtualInst.wfi && FuType.FuTypeOrR(decodedInst.fuType, FuType.csr) && CSROpType.isWfi(
        decodedInst.fuOpType
      ) ||
      io.fromCSR.virtualInst.wrs_nto && FuType.FuTypeOrR(decodedInst.fuType, FuType.csr) && CSROpType.isWrsNto(
        decodedInst.fuOpType
      ) ||
      io.fromCSR.virtualInst.cboZ && isCboZero ||
      io.fromCSR.virtualInst.cboCF && (isCboClean || isCboFlush) ||
      io.fromCSR.virtualInst.cboI && isCboInval

  decodedInst.exceptionVec(illegalInstr) := exceptionII || io.enq.ctrlFlow.exceptionVec(EX_II)
  decodedInst.exceptionVec(virtualInstr) := exceptionVI

  //update exceptionVec: from frontend trigger's breakpoint exception. To reduce 1 bit of overhead in ibuffer entry.
  decodedInst.exceptionVec(breakPoint) := TriggerAction.isExp(ctrl_flow.trigger)

  decodedInst.imm := LookupTree(
    decodedInst.selImm,
    ImmUnion.immSelMap.map {
      x =>
        val minBits = x._2.minBitsFromInstr(ctrl_flow.instr)
        require(minBits.getWidth == x._2.len)
        x._1 -> minBits
    }
  )

  private val isLs     = FuType.isLoadStore(decodedInst.fuType)
  private val isVls    = inst.isVecStore || inst.isVecLoad
  private val isStore  = FuType.isStore(decodedInst.fuType)
  private val isAMO    = FuType.isAMO(decodedInst.fuType)
  private val isVStore = FuType.isVStore(decodedInst.fuType)
  private val isBranch = !decodedInst.preDecodeInfo.notCFI || FuType.isJump(decodedInst.fuType)

  decodedInst.commitType := Cat(isLs | isVls, (isStore && !isAMO) | isVStore | isBranch)

  decodedInst.isVset := FuType.isVset(decodedInst.fuType)

  private val needReverseInsts = Seq(VRSUB_VI, VRSUB_VX, VFRDIV_VF, VFRSUB_VF, VFMV_F_S)
  private val vextInsts        = Seq(VZEXT_VF2, VZEXT_VF4, VZEXT_VF8, VSEXT_VF2, VSEXT_VF4, VSEXT_VF8)

  private val narrowInsts = Seq(
    VNSRA_WV,
    VNSRA_WX,
    VNSRA_WI,
    VNSRL_WV,
    VNSRL_WX,
    VNSRL_WI,
    VNCLIP_WV,
    VNCLIP_WX,
    VNCLIP_WI,
    VNCLIPU_WV,
    VNCLIPU_WX,
    VNCLIPU_WI
  )

  private val maskDstInsts = Seq(
    VMADC_VV,
    VMADC_VX,
    VMADC_VI,
    VMADC_VVM,
    VMADC_VXM,
    VMADC_VIM,
    VMSBC_VV,
    VMSBC_VX,
    VMSBC_VVM,
    VMSBC_VXM,
    VMAND_MM,
    VMNAND_MM,
    VMANDN_MM,
    VMXOR_MM,
    VMOR_MM,
    VMNOR_MM,
    VMORN_MM,
    VMXNOR_MM,
    VMSEQ_VV,
    VMSEQ_VX,
    VMSEQ_VI,
    VMSNE_VV,
    VMSNE_VX,
    VMSNE_VI,
    VMSLE_VV,
    VMSLE_VX,
    VMSLE_VI,
    VMSLEU_VV,
    VMSLEU_VX,
    VMSLEU_VI,
    VMSLT_VV,
    VMSLT_VX,
    VMSLTU_VV,
    VMSLTU_VX,
    VMSGT_VX,
    VMSGT_VI,
    VMSGTU_VX,
    VMSGTU_VI,
    VMFEQ_VV,
    VMFEQ_VF,
    VMFNE_VV,
    VMFNE_VF,
    VMFLT_VV,
    VMFLT_VF,
    VMFLE_VV,
    VMFLE_VF,
    VMFGT_VF,
    VMFGE_VF
  )

  private val maskOpInsts = Seq(
    VMAND_MM,
    VMNAND_MM,
    VMANDN_MM,
    VMXOR_MM,
    VMOR_MM,
    VMNOR_MM,
    VMORN_MM,
    VMXNOR_MM
  )

  private val vmaInsts = Seq(
    VMACC_VV,
    VMACC_VX,
    VNMSAC_VV,
    VNMSAC_VX,
    VMADD_VV,
    VMADD_VX,
    VNMSUB_VV,
    VNMSUB_VX,
    VWMACCU_VV,
    VWMACCU_VX,
    VWMACC_VV,
    VWMACC_VX,
    VWMACCSU_VV,
    VWMACCSU_VX,
    VWMACCUS_VX
  )

  private val wfflagsInsts = Seq(
    // opfff
    FADD_S,
    FSUB_S,
    FADD_D,
    FSUB_D,
    FADD_H,
    FSUB_H,
    FEQ_S,
    FLT_S,
    FLE_S,
    FEQ_D,
    FLT_D,
    FLE_D,
    FEQ_H,
    FLT_H,
    FLE_H,
    FMIN_S,
    FMAX_S,
    FMIN_D,
    FMAX_D,
    FMIN_H,
    FMAX_H,
    FMUL_S,
    FMUL_D,
    FMUL_H,
    FDIV_S,
    FDIV_D,
    FSQRT_S,
    FSQRT_D,
    FDIV_H,
    FSQRT_H,
    FMADD_S,
    FMSUB_S,
    FNMADD_S,
    FNMSUB_S,
    FMADD_D,
    FMSUB_D,
    FNMADD_D,
    FNMSUB_D,
    FMADD_H,
    FMSUB_H,
    FNMADD_H,
    FNMSUB_H,
    FSGNJ_S,
    FSGNJN_S,
    FSGNJX_S,
    FSGNJ_H,
    FSGNJN_H,
    FSGNJX_H,
    // opfvv
    VFADD_VV,
    VFSUB_VV,
    VFWADD_VV,
    VFWSUB_VV,
    VFWADD_WV,
    VFWSUB_WV,
    VFMUL_VV,
    VFDIV_VV,
    VFWMUL_VV,
    VFMACC_VV,
    VFNMACC_VV,
    VFMSAC_VV,
    VFNMSAC_VV,
    VFMADD_VV,
    VFNMADD_VV,
    VFMSUB_VV,
    VFNMSUB_VV,
    VFWMACC_VV,
    VFWNMACC_VV,
    VFWMSAC_VV,
    VFWNMSAC_VV,
    VFSQRT_V,
    VFMIN_VV,
    VFMAX_VV,
    VMFEQ_VV,
    VMFNE_VV,
    VMFLT_VV,
    VMFLE_VV,
    VFSGNJ_VV,
    VFSGNJN_VV,
    VFSGNJX_VV,
    // opfvf
    VFADD_VF,
    VFSUB_VF,
    VFRSUB_VF,
    VFWADD_VF,
    VFWSUB_VF,
    VFWADD_WF,
    VFWSUB_WF,
    VFMUL_VF,
    VFDIV_VF,
    VFRDIV_VF,
    VFWMUL_VF,
    VFMACC_VF,
    VFNMACC_VF,
    VFMSAC_VF,
    VFNMSAC_VF,
    VFMADD_VF,
    VFNMADD_VF,
    VFMSUB_VF,
    VFNMSUB_VF,
    VFWMACC_VF,
    VFWNMACC_VF,
    VFWMSAC_VF,
    VFWNMSAC_VF,
    VFMIN_VF,
    VFMAX_VF,
    VMFEQ_VF,
    VMFNE_VF,
    VMFLT_VF,
    VMFLE_VF,
    VMFGT_VF,
    VMFGE_VF,
    VFSGNJ_VF,
    VFSGNJN_VF,
    VFSGNJX_VF,
    // vfred
    VFREDOSUM_VS,
    VFREDUSUM_VS,
    VFREDMAX_VS,
    VFREDMIN_VS,
    VFWREDOSUM_VS,
    VFWREDUSUM_VS,
    // fcvt & vfcvt
    FCVT_S_W,
    FCVT_S_WU,
    FCVT_S_L,
    FCVT_S_LU,
    FCVT_W_S,
    FCVT_WU_S,
    FCVT_L_S,
    FCVT_LU_S,
    FCVT_D_W,
    FCVT_D_WU,
    FCVT_D_L,
    FCVT_D_LU,
    FCVT_W_D,
    FCVT_WU_D,
    FCVT_L_D,
    FCVT_LU_D,
    FCVT_S_D,
    FCVT_D_S,
    FCVT_S_H,
    FCVT_H_S,
    FCVT_H_D,
    FCVT_D_H,
    FCVT_H_W,
    FCVT_H_WU,
    FCVT_H_L,
    FCVT_H_LU,
    FCVT_W_H,
    FCVT_WU_H,
    FCVT_L_H,
    FCVT_LU_H,
    VFCVT_XU_F_V,
    VFCVT_X_F_V,
    VFCVT_RTZ_XU_F_V,
    VFCVT_RTZ_X_F_V,
    VFCVT_F_XU_V,
    VFCVT_F_X_V,
    VFWCVT_XU_F_V,
    VFWCVT_X_F_V,
    VFWCVT_RTZ_XU_F_V,
    VFWCVT_RTZ_X_F_V,
    VFWCVT_F_XU_V,
    VFWCVT_F_X_V,
    VFWCVT_F_F_V,
    VFNCVT_XU_F_W,
    VFNCVT_X_F_W,
    VFNCVT_RTZ_XU_F_W,
    VFNCVT_RTZ_X_F_W,
    VFNCVT_F_XU_W,
    VFNCVT_F_X_W,
    VFNCVT_F_F_W,
    VFNCVT_ROD_F_F_W,
    VFRSQRT7_V,
    VFREC7_V,
    // zfa
    FLEQ_H,
    FLEQ_S,
    FLEQ_D,
    FLTQ_H,
    FLTQ_S,
    FLTQ_D,
    FMINM_H,
    FMINM_S,
    FMINM_D,
    FMAXM_H,
    FMAXM_S,
    FMAXM_D,
    FROUND_H,
    FROUND_S,
    FROUND_D,
    FROUNDNX_H,
    FROUNDNX_S,
    FROUNDNX_D,
    FCVTMOD_W_D
  )

  private val scalaNeedFrmInsts = Seq(
    FADD_S,
    FSUB_S,
    FADD_D,
    FSUB_D,
    FADD_H,
    FSUB_H,
    FCVT_W_S,
    FCVT_WU_S,
    FCVT_L_S,
    FCVT_LU_S,
    FCVT_W_D,
    FCVT_WU_D,
    FCVT_L_D,
    FCVT_LU_D,
    FCVT_S_D,
    FCVT_D_S,
    FCVT_W_H,
    FCVT_WU_H,
    FCVT_L_H,
    FCVT_LU_H,
    FCVT_S_H,
    FCVT_H_S,
    FCVT_H_D,
    FCVT_D_H,
    FROUND_H,
    FROUND_S,
    FROUND_D,
    FROUNDNX_H,
    FROUNDNX_S,
    FROUNDNX_D
  )

  private val vectorNeedFrmInsts = Seq(
    VFSLIDE1UP_VF,
    VFSLIDE1DOWN_VF
  )

  decodedInst.wfflags               := wfflagsInsts.map(_ === inst.ALL).reduce(_ || _)
  decodedInst.needFrm.scalaNeedFrm  := scalaNeedFrmInsts.map(_ === inst.ALL).reduce(_ || _)
  decodedInst.needFrm.vectorNeedFrm := vectorNeedFrmInsts.map(_ === inst.ALL).reduce(_ || _)
  decodedInst.vpu                   := 0.U.asTypeOf(decodedInst.vpu) // Todo: Connect vpu decoder
  decodedInst.vpu.vill              := io.enq.vtype.illegal
  decodedInst.vpu.vma               := io.enq.vtype.vma
  decodedInst.vpu.vta               := io.enq.vtype.vta
  decodedInst.vpu.vsew              := io.enq.vtype.vsew
  decodedInst.vpu.vlmul             := io.enq.vtype.vlmul
  decodedInst.vpu.vm                := inst.VM
  decodedInst.vpu.nf                := inst.NF
  decodedInst.vpu.veew              := inst.WIDTH
  decodedInst.vpu.isReverse         := needReverseInsts.map(_ === inst.ALL).reduce(_ || _)
  decodedInst.vpu.isExt             := vextInsts.map(_ === inst.ALL).reduce(_ || _)
  val isNarrow        = narrowInsts.map(_ === inst.ALL).reduce(_ || _)
  val isDstMask       = maskDstInsts.map(_ === inst.ALL).reduce(_ || _)
  val isOpMask        = maskOpInsts.map(_ === inst.ALL).reduce(_ || _)
  val isVload         = FuType.isVLoad(decodedInst.fuType)
  val isVlx           = isVload && (decodedInst.fuOpType === VlduType.vloxe || decodedInst.fuOpType === VlduType.vluxe)
  val isVle           =
    isVload && (decodedInst.fuOpType === VlduType.vle || decodedInst.fuOpType === VlduType.vleff || decodedInst.fuOpType === VlduType.vlse)
  val isVlm           = isVload && (decodedInst.fuOpType === VlduType.vlm)
  val isFof           = isVload && (decodedInst.fuOpType === VlduType.vleff)
  val isWritePartVd   =
    decodedInst.uopSplitType === UopSplitType.VEC_VRED || decodedInst.uopSplitType === UopSplitType.VEC_0XV || decodedInst.uopSplitType === UopSplitType.VEC_VWW
  val isVma           = vmaInsts.map(_ === inst.ALL).reduce(_ || _)
  val emulIsFrac      =
    Cat(~decodedInst.vpu.vlmul(2), decodedInst.vpu.vlmul(1, 0)) +& decodedInst.vpu.veew < 4.U +& decodedInst.vpu.vsew
  val vstartIsNotZero = io.enq.vstart =/= 0.U
  decodedInst.vpu.isNarrow      := isNarrow
  decodedInst.vpu.isDstMask     := isDstMask
  decodedInst.vpu.isOpMask      := isOpMask
  decodedInst.vpu.isDependOldVd := isVppu || isVecOPF || isVStore || (isDstMask && !isOpMask) || isNarrow || isVlx || isVma || isFof || vstartIsNotZero
  decodedInst.vpu.isWritePartVd := isWritePartVd || isVlm || isVle && emulIsFrac
  decodedInst.vpu.vstart        := io.enq.vstart
  decodedInst.vpu.isVleff       := isFof && inst.NF === 0.U
  decodedInst.vpu.specVill      := io.enq.vtype.illegal
  decodedInst.vpu.specVma       := io.enq.vtype.vma
  decodedInst.vpu.specVta       := io.enq.vtype.vta
  decodedInst.vpu.specVsew      := io.enq.vtype.vsew
  decodedInst.vpu.specVlmul     := io.enq.vtype.vlmul

  decodedInst.vlsInstr := isVls

  decodedInst.srcType(3) := Mux(inst.VM === 0.U, SrcType.vp, SrcType.DC) // mask src
  decodedInst.srcType(4) := SrcType.vp                                   // vconfig

  val uopInfoGen = Module(new UopInfoGen)
  uopInfoGen.io.in.preInfo.isVecArith := inst.isVecArith
  uopInfoGen.io.in.preInfo.isVecMem   := inst.isVecStore || inst.isVecLoad
  uopInfoGen.io.in.preInfo.isAmoCAS   := inst.isAMOCAS

  uopInfoGen.io.in.preInfo.typeOfSplit := decodedInst.uopSplitType
  uopInfoGen.io.in.preInfo.vsew        := decodedInst.vpu.vsew
  uopInfoGen.io.in.preInfo.vlmul       := decodedInst.vpu.vlmul
  uopInfoGen.io.in.preInfo.vwidth      := inst.RM
  uopInfoGen.io.in.preInfo.vmvn        := inst.IMM5_OPIVI(2, 0)
  uopInfoGen.io.in.preInfo.nf          := inst.NF
  uopInfoGen.io.in.preInfo.isVlsr      := decodedInst.fuOpType === VlduType.vlr || decodedInst.fuOpType === VstuType.vsr
  uopInfoGen.io.in.preInfo.isVlsm      := decodedInst.fuOpType === VlduType.vlm || decodedInst.fuOpType === VstuType.vsm
  io.deq.isComplex                     := uopInfoGen.io.out.isComplex
  io.deq.uopInfo.numOfUop              := uopInfoGen.io.out.uopInfo.numOfUop
  io.deq.uopInfo.numOfWB               := uopInfoGen.io.out.uopInfo.numOfWB
  io.deq.uopInfo.lmul                  := uopInfoGen.io.out.uopInfo.lmul

  // decode for SoftPrefetch instructions (prefetch.w / prefetch.r / prefetch.i)
  val isSoftPrefetch = inst.OPCODE === BitPat("b0010011") && inst.FUNCT3 === BitPat("b110") && inst.RD === 0.U
  val isPreW         = isSoftPrefetch && inst.RS2 === 3.U(5.W)
  val isPreR         = isSoftPrefetch && inst.RS2 === 1.U(5.W)
  val isPreI         = isSoftPrefetch && inst.RS2 === 0.U(5.W)

  // for fli.s|fli.d instruction
  val isFLI =
    inst.FUNCT7 === BitPat("b11110??") && inst.RS2 === 1.U && inst.RM === 0.U && inst.OPCODE5Bit === OPCODE5Bit.OP_FP

  when(isCsrrVl) {
    // convert to vsetvl instruction
    decodedInst.srcType(0)    := SrcType.no
    decodedInst.srcType(1)    := SrcType.no
    decodedInst.srcType(2)    := SrcType.no
    decodedInst.srcType(3)    := SrcType.no
    decodedInst.srcType(4)    := SrcType.vp
    decodedInst.lsrc(4)       := Vl_IDX.U
    decodedInst.waitForward   := false.B
    decodedInst.blockBackward := false.B
  }.elsewhen(isCsrrVlenb) {
    // convert to addi instruction
    decodedInst.srcType(0)     := SrcType.reg
    decodedInst.srcType(1)     := SrcType.imm
    decodedInst.srcType(2)     := SrcType.no
    decodedInst.srcType(3)     := SrcType.no
    decodedInst.srcType(4)     := SrcType.no
    decodedInst.selImm         := SelImm.IMM_I
    decodedInst.waitForward    := false.B
    decodedInst.blockBackward  := false.B
    decodedInst.canRobCompress := true.B
  }.elsewhen(isPreW || isPreR || isPreI) {
    decodedInst.selImm         := SelImm.IMM_S
    decodedInst.fuType         := FuType.ldu.U
    decodedInst.canRobCompress := false.B
  }.elsewhen(isZimop) {
    // set srcType for zimop
    decodedInst.srcType(0) := SrcType.reg
    decodedInst.srcType(1) := SrcType.imm
    // use x0 as src1
    decodedInst.lsrc(0)    := 0.U
  }

  io.deq.decodedInst        := decodedInst
  io.deq.decodedInst.rfWen  := (decodedInst.ldest =/= 0.U) && decodedInst.rfWen
  io.deq.decodedInst.fuType := Mux1H(Seq(
    // keep condition
    (!FuType.FuTypeOrR(decodedInst.fuType, FuType.vldu, FuType.vstu) && !isCsrrVl && !isCsrrVlenb)         -> decodedInst.fuType,
    isCsrrVl                                                                                               -> FuType.vsetfwf.U,
    isCsrrVlenb                                                                                            -> FuType.alu.U,
    // change vlsu to vseglsu when NF =/= 0.U
    (FuType.FuTypeOrR(
      decodedInst.fuType,
      FuType.vldu,
      FuType.vstu
    ) && inst.NF === 0.U || (inst.NF =/= 0.U && (inst.MOP === "b00".U && inst.SUMOP === "b01000".U)))      -> decodedInst.fuType,
    // MOP === b00 && SUMOP === b01000: unit-stride whole register store
    // MOP =/= b00                    : strided and indexed store
    (FuType.FuTypeOrR(
      decodedInst.fuType,
      FuType.vstu
    ) && inst.NF =/= 0.U && ((inst.MOP === "b00".U && inst.SUMOP =/= "b01000".U) || inst.MOP =/= "b00".U)) -> FuType.vsegstu.U,
    // MOP === b00 && LUMOP === b01000: unit-stride whole register load
    // MOP =/= b00                    : strided and indexed load
    (FuType.FuTypeOrR(
      decodedInst.fuType,
      FuType.vldu
    ) && inst.NF =/= 0.U && ((inst.MOP === "b00".U && inst.LUMOP =/= "b01000".U) || inst.MOP =/= "b00".U)) -> FuType.vsegldu.U
  ))
  io.deq.decodedInst.imm    := MuxCase(
    decodedInst.imm,
    Seq(
      isCsrrVlenb -> (VLEN / 8).U,
      isZimop     -> 0.U
    )
  )

  io.deq.decodedInst.fuOpType := MuxCase(
    decodedInst.fuOpType,
    Seq(
      isCsrrVl                                  -> VSETOpType.csrrvl,
      isCsrrVlenb                               -> ALUOpType.add,
      isFLI                                     -> Cat(1.U, inst.FMT, inst.RS1),
      (isPreW || isPreR || isPreI)              -> Mux1H(Seq(
        isPreW -> LSUOpType.prefetch_w,
        isPreR -> LSUOpType.prefetch_r,
        isPreI -> LSUOpType.prefetch_i
      )),
      (isCboInval && io.fromCSR.special.cboI2F) -> LSUOpType.cbo_flush
    )
  )

  // Don't compress in the same Rob entry when crossing Ftq entry boundary
  io.deq.decodedInst.canRobCompress := decodedInst.canRobCompress && !io.enq.ctrlFlow.isLastInFtqEntry

  //-------------------------------------------------------------
  // Debug Info
//  XSDebug("in:  instr=%x pc=%x excepVec=%b crossPageIPFFix=%d\n",
//    io.enq.ctrl_flow.instr, io.enq.ctrl_flow.pc, io.enq.ctrl_flow.exceptionVec.asUInt,
//    io.enq.ctrl_flow.crossPageIPFFix)
// XSDebug("out: srcType(0)=%b srcType(1)=%b srcType(2)=%b lsrc(0)=%d lsrc(1)=%d lsrc(2)=%d ldest=%d fuType=%b
  // fuOpType=%b\n",
//    io.deq.cf_ctrl.ctrl.srcType(0), io.deq.cf_ctrl.ctrl.srcType(1), io.deq.cf_ctrl.ctrl.srcType(2),
//    io.deq.cf_ctrl.ctrl.lsrc(0), io.deq.cf_ctrl.ctrl.lsrc(1), io.deq.cf_ctrl.ctrl.lsrc(2),
//    io.deq.cf_ctrl.ctrl.ldest, io.deq.cf_ctrl.ctrl.fuType, io.deq.cf_ctrl.ctrl.fuOpType)
//  XSDebug("out: rfWen=%d fpWen=%d isXSTrap=%d noSpecExec=%d isBlocked=%d flushPipe=%d imm=%x\n",
//    io.deq.cf_ctrl.ctrl.rfWen, io.deq.cf_ctrl.ctrl.fpWen, io.deq.cf_ctrl.ctrl.isXSTrap,
//    io.deq.cf_ctrl.ctrl.noSpecExec, io.deq.cf_ctrl.ctrl.blockBackward, io.deq.cf_ctrl.ctrl.flushPipe,
//    io.deq.cf_ctrl.ctrl.imm)
//  XSDebug("out: excepVec=%b\n", io.deq.cf_ctrl.cf.exceptionVec.asUInt)
}
