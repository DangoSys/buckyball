// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2022 Jiuyang Liu <liu@jiuyang.me>
// Port to buckyball framework

package framework.gpdomain.sequencer

import chisel3._
import chisel3.util._
import chisel3.util.experimental.decode.DecodeBundle
import framework.gpdomain.sequencer.decoder.{Decoder, DomainDecoderParameter}

/** CSR Interface from Scalar Core */
class CSRInterface(vlWidth: Int) extends Bundle {

  /** Vector Length Register `vl` */
  val vl: UInt = UInt(vlWidth.W)

  /** Vector Start Index CSR `vstart` */
  val vStart: UInt = UInt(vlWidth.W)

  /** Vector Register Grouping `vlmul[2:0]` subfield of `vtype` */
  val vlmul: UInt = UInt(3.W)

  /** Vector Selected Element Width `vsew[2:0]` subfield of `vtype` */
  val vSew: UInt = UInt(2.W)

  /** Vector Fixed-Point Rounding Mode Register `vxrm` */
  val vxrm: UInt = UInt(2.W)

  /** Floating-point rounding mode */
  val frm: UInt = UInt(3.W)

  /** Vector Tail Agnostic */
  val vta: Bool = Bool()

  /** Vector Mask Agnostic */
  val vma: Bool = Bool()
}

/** Instruction record for tracking issued instructions */
class InstructionRecord(instructionIndexWidth: Int) extends Bundle {

  /** Index of this instruction, maintained by instruction counter */
  val instructionIndex: UInt = UInt(instructionIndexWidth.W)

  /** Whether instruction is load/store */
  val isLoadStore: Bool = Bool()

  /** Whether instruction is mask type */
  val maskType: Bool = Bool()

  val gather: Bool = Bool()
  val pop:    Bool = Bool()
}

/** Instruction execution state */
class InstructionState extends Bundle {

  /** Wait for last signal from each lane */
  val wLast: Bool = Bool()

  /** The slot is idle */
  val idle: Bool = Bool()

  /** Used for mask unit, schedule mask unit to execute */
  val wMaskUnitLast: Bool = Bool()

  /** Used for instruction commit */
  val sCommit: Bool = Bool()
}

/** Instruction control state for each slot */
class InstructionControl(instIndexWidth: Int, laneSize: Int) extends Bundle {

  /** Metadata for this instruction */
  val record: InstructionRecord = new InstructionRecord(instIndexWidth)

  /** Control state to record the current execution state */
  val state: InstructionState = new InstructionState

  /** Tag for recording each lane is finished for this instruction */
  val endTag: Vec[Bool] = Vec(laneSize + 1, Bool())

  /** Fixed-point saturation flag */
  val vxsat: Bool = Bool()
}

/** Issue token for token manager */
class IssueToken(instructionIndexBits: Int) extends Bundle {
  val instructionIndex: UInt = UInt(instructionIndexBits.W)
  val writeV0:          Bool = Bool()
  val useV0AsMask:      Bool = Bool()
  val isLoadStore:      Bool = Bool()
  val toLane:           Bool = Bool()
  val toMask:           Bool = Bool()
}

/** Instruction pipe bundle for queuing */
class InstructionPipeBundle(
  xLen:                 Int,
  vLen:                 Int,
  instructionIndexBits: Int,
  vlMaxBits:            Int)
    extends Bundle {
  val instruction:      UInt         = UInt(32.W)
  val rs1Data:          UInt         = UInt(xLen.W)
  val rs2Data:          UInt         = UInt(xLen.W)
  val vl:               UInt         = UInt(32.W)
  val vstart:           UInt         = UInt(32.W)
  val vtype:            UInt         = UInt(32.W)
  val decodeResult:     DecodeBundle = Decoder.bundle(DomainDecoderParameter.decoderParam).cloneType
  val instructionIndex: UInt         = UInt(instructionIndexBits.W)
  val vdIsV0:           Bool         = Bool()
  val writeByte:        UInt         = UInt(vlMaxBits.W)
}

// ============================================================================
// VRF Related Bundles
// ============================================================================

/** Request to access VRF in each lanes */
class VRFReadRequest(regNumBits: Int, offsetBits: Int, instructionIndexBits: Int) extends Bundle {

  /** address to access VRF (v0, v1, v2, ...) */
  val vs: UInt = UInt(regNumBits.W)

  /** read vs1 vs2 vd? */
  val readSource: UInt = UInt(2.W)

  /** the offset of VRF access */
  val offset: UInt = UInt(offsetBits.W)

  /** index for record the age of instruction, designed for handling RAW hazard */
  val instructionIndex: UInt = UInt(instructionIndexBits.W)
}

class VRFWriteRequest(
  regNumBits:           Int,
  offsetBits:           Int,
  instructionIndexSize: Int,
  dataPathWidth:        Int)
    extends Bundle {

  /** address to access VRF (v0, v1, v2, ...) */
  val vd: UInt = UInt(regNumBits.W)

  /** the offset of VRF access */
  val offset: UInt = UInt(offsetBits.W)

  /** write mask in byte */
  val mask: UInt = UInt((dataPathWidth / 8).W)

  /** data to write to VRF */
  val data: UInt = UInt(dataPathWidth.W)

  /** this is the last write of this instruction */
  val last: Bool = Bool()

  /** used to update the record in VRF */
  val instructionIndex: UInt = UInt(instructionIndexSize.W)
}

class V0Update(datapathWidth: Int, vrfOffsetBits: Int) extends Bundle {
  val data:   UInt = UInt(datapathWidth.W)
  val offset: UInt = UInt(vrfOffsetBits.W)
  val mask:   UInt = UInt((datapathWidth / 8).W)
}

// ============================================================================
// Lane Related Bundles
// ============================================================================

/** Request from sequencer to lane */
class LaneRequest(
  instructionIndexBits: Int,
  datapathWidth:        Int,
  vlMaxBits:            Int,
  laneNumber:           Int,
  dataPathByteWidth:    Int)
    extends Bundle {
  val instructionIndex: UInt = UInt(instructionIndexBits.W)

  // decode
  val decodeResult: DecodeBundle = Decoder.bundle(DomainDecoderParameter.decoderParam).cloneType
  val loadStore:    Bool         = Bool()
  val issueInst:    Bool         = Bool()
  val store:        Bool         = Bool()
  val special:      Bool         = Bool()
  val lsWholeReg:   Bool         = Bool()

  // instruction
  val vs1: UInt = UInt(5.W)
  val vs2: UInt = UInt(5.W)
  val vd:  UInt = UInt(5.W)

  val loadStoreEEW: UInt = UInt(2.W)
  val mask:         Bool = Bool()
  val segment:      UInt = UInt(3.W)

  /** data of rs1 */
  val readFromScalar: UInt = UInt(datapathWidth.W)

  val csrInterface: CSRInterface = new CSRInterface(vlMaxBits)

  val writeCount: UInt = UInt((vlMaxBits - log2Ceil(laneNumber) - log2Ceil(dataPathByteWidth)).W)

  val maskE0: Bool = Bool()
}

class LaneResponse(chaining1HBits: Int, vlMaxBits: Int) extends Bundle {
  val instructionFinished: UInt = UInt(chaining1HBits.W)
  val vxsatReport:         UInt = UInt(chaining1HBits.W)
  val popCount:            UInt = UInt(vlMaxBits.W)
}

class LaneResponseFeedback(instructionIndexBits: Int) extends Bundle {

  /** which instruction is the source of this transaction */
  val instructionIndex: UInt = UInt(instructionIndexBits.W)

  /** for instructions that might finish in other lanes */
  val complete: Bool = Bool()
}

// ============================================================================
// Mask Related Bundles
// ============================================================================

class MaskRequest(maskGroupSizeBits: Int) extends Bundle {

  /** select which mask group */
  val maskSelect: UInt = UInt(maskGroupSizeBits.W)

  /** The sew of instruction which is requesting for mask */
  val maskSelectSew: UInt = UInt(2.W)

  val slide: Bool = Bool()
}

class MaskRequestAck(maskGroupWidth: Int) extends Bundle {
  val data: UInt = UInt(maskGroupWidth.W)
}

class MaskUnitExeReq(
  eLen:                 Int,
  datapathWidth:        Int,
  instructionIndexBits: Int,
  fpuEnable:            Boolean)
    extends Bundle {
  val source1:          UInt         = UInt(datapathWidth.W)
  val source2:          UInt         = UInt(datapathWidth.W)
  val index:            UInt         = UInt(instructionIndexBits.W)
  val ffo:              UInt         = UInt((datapathWidth / eLen).W)
  val fpReduceValid:    Option[UInt] = Option.when(fpuEnable)(UInt((datapathWidth / eLen).W))
  val maskRequestToLSU: Bool         = Bool()
}

// ============================================================================
// LSU Related Bundles
// ============================================================================

class LSUInstructionInformation extends Bundle {
  val nf:              UInt = UInt(3.W)
  val mew:             Bool = Bool()
  val mop:             UInt = UInt(2.W)
  val lumop:           UInt = UInt(5.W)
  val eew:             UInt = UInt(2.W)
  val vs3:             UInt = UInt(5.W)
  val isStore:         Bool = Bool()
  val maskedLoadStore: Bool = Bool()

  def fof: Bool = mop === 0.U && lumop(4) && !isStore
}

class LSURequest(dataWidth: Int, chainingSize: Int) extends Bundle {
  val instructionInformation: LSUInstructionInformation = new LSUInstructionInformation
  val rs1Data:                UInt                      = UInt(dataWidth.W)
  val rs2Data:                UInt                      = UInt(dataWidth.W)
  val instructionIndex:       UInt                      = UInt((log2Ceil(chainingSize) + 1).W)
}

class LSURequestInterface(dataWidth: Int, chainingSize: Int, vlWidth: Int) extends Bundle {
  val request:      LSURequest   = new LSURequest(dataWidth, chainingSize)
  val csrInterface: CSRInterface = new CSRInterface(vlWidth)
}

// ============================================================================
// Common Bundles
// ============================================================================

class LastReportBundle(chaining1HBits: Int) extends Bundle {
  val last: UInt = UInt(chaining1HBits.W)
}

class WriteCountReport(vLen: Int, laneNumber: Int, instSize: Int) extends Bundle {
  val count:            UInt = UInt(log2Ceil(vLen / laneNumber).W)
  val instructionIndex: UInt = UInt(instSize.W)
}

final class EmptyBundle extends Bundle
