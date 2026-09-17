package memcore.bus.chi

import chisel3._
import chisel3.util.Cat

// AMBA CHI Issue B, without RSVDC, DataCheck or Poison extensions.
// Field order/encodings checked against OpenXiangShan/OpenNCB's CHI definitions:
// https://github.com/OpenXiangShan/OpenNCB/tree/9a83eb7e6c0e36e0d6b72642ff93641ec65819cf/src/main/scala/openncb/chi
case class ChiParams(nodeIdBits: Int = 7, addressBits: Int = 44, dataBits: Int = 256) {
  require(nodeIdBits >= 7 && nodeIdBits <= 11)
  require(addressBits >= 44 && addressBits <= 52)
  require(Set(128, 256, 512).contains(dataBits))
  val beatsPerLine = 512 / dataBits
  val bytesPerBeat = dataBits / 8
}

abstract class ChiFlit extends Bundle {
  // Explicit wire packing: QoS is at bit zero, independent of Bundle.asUInt order.
  def fieldsLSB: Seq[UInt]
  final def packed: UInt = Cat(fieldsLSB.reverse)

  final def unpack(bits: UInt): Unit = {
    require(bits.getWidth == fieldsLSB.map(_.getWidth).sum)
    var offset = 0
    for (field <- fieldsLSB) {
      field := bits(offset + field.getWidth - 1, offset)
      offset += field.getWidth
    }
  }

  final def flitWidth: Int = fieldsLSB.map(_.getWidth).sum
}

class ChiReq(p: ChiParams) extends ChiFlit {
  val qos                 = UInt(4.W)
  val tgtId               = UInt(p.nodeIdBits.W)
  val srcId               = UInt(p.nodeIdBits.W)
  val txnId               = UInt(8.W)
  val returnNid           = UInt(p.nodeIdBits.W)
  val stashNidValidEndian = UInt(1.W)
  val returnTxnId         = UInt(8.W)
  val opcode              = UInt(6.W)
  val size                = UInt(3.W)
  val addr                = UInt(p.addressBits.W)
  val ns                  = UInt(1.W)
  val likelyShared        = UInt(1.W)
  val allowRetry          = UInt(1.W)
  val order               = UInt(2.W)
  val pCrdType            = UInt(4.W)
  val memAttr             = UInt(4.W)
  val snpAttr             = UInt(1.W)
  val lpid                = UInt(5.W)
  val exclSnoopMe         = UInt(1.W)
  val expCompAck          = UInt(1.W)
  val traceTag            = UInt(1.W)

  def fieldsLSB = Seq(
    qos,
    tgtId,
    srcId,
    txnId,
    returnNid,
    stashNidValidEndian,
    returnTxnId,
    opcode,
    size,
    addr,
    ns,
    likelyShared,
    allowRetry,
    order,
    pCrdType,
    memAttr,
    snpAttr,
    lpid,
    exclSnoopMe,
    expCompAck,
    traceTag
  )

}

class ChiRsp(p: ChiParams) extends ChiFlit {
  val qos       = UInt(4.W)
  val tgtId     = UInt(p.nodeIdBits.W)
  val srcId     = UInt(p.nodeIdBits.W)
  val txnId     = UInt(8.W)
  val opcode    = UInt(4.W)
  val respErr   = UInt(2.W)
  val resp      = UInt(3.W)
  val fwdState  = UInt(3.W)
  val dbid      = UInt(8.W)
  val pCrdType  = UInt(4.W)
  val traceTag  = UInt(1.W)
  def fieldsLSB = Seq(qos, tgtId, srcId, txnId, opcode, respErr, resp, fwdState, dbid, pCrdType, traceTag)
}

class ChiDat(p: ChiParams) extends ChiFlit {
  val qos       = UInt(4.W)
  val tgtId     = UInt(p.nodeIdBits.W)
  val srcId     = UInt(p.nodeIdBits.W)
  val txnId     = UInt(8.W)
  val homeNid   = UInt(p.nodeIdBits.W)
  val opcode    = UInt(3.W)
  val respErr   = UInt(2.W)
  val resp      = UInt(3.W)
  val fwdState  = UInt(3.W)
  val dbid      = UInt(8.W)
  val ccid      = UInt(2.W)
  val dataId    = UInt(2.W)
  val traceTag  = UInt(1.W)
  val be        = UInt(p.bytesPerBeat.W)
  val data      = UInt(p.dataBits.W)
  def fieldsLSB =
    Seq(qos, tgtId, srcId, txnId, homeNid, opcode, respErr, resp, fwdState, dbid, ccid, dataId, traceTag, be, data)
}

class ChiSnp(p: ChiParams) extends ChiFlit {
  val qos         = UInt(4.W)
  val srcId       = UInt(p.nodeIdBits.W)
  val txnId       = UInt(8.W)
  val fwdNid      = UInt(p.nodeIdBits.W)
  val fwdTxnId    = UInt(8.W)
  val opcode      = UInt(5.W)
  val addr        = UInt((p.addressBits - 3).W)
  val ns          = UInt(1.W)
  val doNotGoToSd = UInt(1.W)
  val retToSrc    = UInt(1.W)
  val traceTag    = UInt(1.W)
  def fieldsLSB   = Seq(qos, srcId, txnId, fwdNid, fwdTxnId, opcode, addr, ns, doNotGoToSd, retToSrc, traceTag)
}

object ChiOpcode {
  val ReadShared         = 0x01
  val ReadUnique         = 0x07
  val CleanInvalid       = 0x09
  val ReadNotSharedDirty = 0x26
  val Evict              = 0x0d
  val WriteBackFull      = 0x1b
  val SnpResp            = 0x01
  val CompAck            = 0x02
  val RetryAck           = 0x03
  val PCrdGrant          = 0x07
  val CompDBIDResp       = 0x05
  val SnpNotSharedDirty  = 0x04
  val SnpUnique          = 0x07
  val SnpCleanInvalid    = 0x09
  val SnpRespData        = 0x01
  val CopyBackWrData     = 0x02
  val ReadNoSnp          = 0x04
  val WriteNoSnpPtl      = 0x1c
  val WriteNoSnpFull     = 0x1d
  val Comp               = 0x04
  val DBIDResp           = 0x06
  val NonCopyBackWrData  = 0x03
  val CompData           = 0x04
}
