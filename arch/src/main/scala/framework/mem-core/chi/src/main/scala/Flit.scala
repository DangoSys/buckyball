package memcore.bus.chi

import chisel3._
import chisel3.util.Cat

// AMBA CHI Issue H base profile. Optional buses default to absent.
case class Params(
  nodeIdBits:  Int = 7,
  addressBits: Int = 44,
  dataBits:    Int = 256) {
  require(nodeIdBits >= 7 && nodeIdBits <= 16)
  require(addressBits >= 44 && addressBits <= 52)
  require(Set(128, 256, 512).contains(dataBits))

  val txnIdBits       = 12
  val dbIdBits        = 12
  val beatsPerLine    = 512 / dataBits
  val bytesPerBeat    = dataBits / 8
  val tagBits         = dataBits / 32
  val tagUpdateBits   = dataBits / 128
  val reqRsvdcBits    = 0
  val datRsvdcBits    = 0
  val mpamBits        = 0
  val pbhaBits        = 0
  val mecidBits       = 0
  val streamIdBits    = 0
  val secSidBits      = 0
  val dataCheckBits   = 0
  val poisonBits      = 0
  val commonSecIdBits = 0
}

abstract class Flit extends Bundle {
  def fieldsLSB: Seq[UInt]

  final def packed: UInt = Cat(fieldsLSB.reverse)

  final def unpack(bits: UInt): Unit = {
    require(bits.getWidth == fieldsLSB.map(_.getWidth).sum)
    var offset = 0
    for (field <- fieldsLSB) {
      if (field.getWidth > 0) {
        field := bits(offset + field.getWidth - 1, offset)
      } else {
        field := 0.U(0.W)
      }
      offset += field.getWidth
    }
  }

  final def flitWidth: Int = fieldsLSB.map(_.getWidth).sum
}

class RequestFlit(p: Params) extends Flit {
  val qos                 = UInt(4.W)
  val tgtId               = UInt(p.nodeIdBits.W)
  val srcId               = UInt(p.nodeIdBits.W)
  val txnId               = UInt(p.txnIdBits.W)
  val returnNid           = UInt(p.nodeIdBits.W)
  val stashNidValidEndian = UInt(1.W)
  val returnTxnId         = UInt(p.txnIdBits.W)
  val opcode              = UInt(7.W)
  val multiReq            = UInt(1.W)
  val size                = UInt(6.W)
  val addr                = UInt(p.addressBits.W)
  val pas                 = UInt(3.W)
  val likelyShared        = UInt(1.W)
  val allowRetry          = UInt(1.W)
  val order               = UInt(2.W)
  val pCrdType            = UInt(4.W)
  val memAttr             = UInt(4.W)
  val snpAttr             = UInt(1.W)
  val lpid                = UInt(8.W)
  val exclSnoopMe         = UInt(1.W)
  val expCompAck          = UInt(1.W)
  val tagOp               = UInt(2.W)
  val traceTag            = UInt(1.W)
  val mpam                = UInt(p.mpamBits.W)
  val pbha                = UInt(p.pbhaBits.W)
  val mecidOrStreamId     = UInt(p.commonSecIdBits.W)
  val secSid              = UInt(p.secSidBits.W)
  val rsvdc               = UInt(p.reqRsvdcBits.W)

  def fieldsLSB = Seq(
    qos,
    tgtId,
    srcId,
    txnId,
    returnNid,
    stashNidValidEndian,
    returnTxnId,
    opcode,
    multiReq,
    size,
    addr,
    pas,
    likelyShared,
    allowRetry,
    order,
    pCrdType,
    memAttr,
    snpAttr,
    lpid,
    exclSnoopMe,
    expCompAck,
    tagOp,
    traceTag,
    mpam,
    pbha,
    mecidOrStreamId,
    secSid,
    rsvdc
  )

}

class ResponseFlit(p: Params) extends Flit {
  val qos         = UInt(4.W)
  val tgtId       = UInt(p.nodeIdBits.W)
  val srcId       = UInt(p.nodeIdBits.W)
  val txnId       = UInt(p.txnIdBits.W)
  val opcode      = UInt(5.W)
  val respErr     = UInt(2.W)
  val resp        = UInt(3.W)
  val fwdState    = UInt(3.W)
  val cBusy       = UInt(3.W)
  val dbid        = UInt(p.dbIdBits.W)
  val pCrdType    = UInt(4.W)
  val tagOp       = UInt(2.W)
  val traceTag    = UInt(1.W)
  val cacheLineId = UInt(6.W)

  def fieldsLSB =
    Seq(qos, tgtId, srcId, txnId, opcode, respErr, resp, fwdState, cBusy, dbid, pCrdType, tagOp, traceTag, cacheLineId)
}

class DataFlit(p: Params) extends Flit {
  val qos         = UInt(4.W)
  val tgtId       = UInt(p.nodeIdBits.W)
  val srcId       = UInt(p.nodeIdBits.W)
  val txnId       = UInt(p.txnIdBits.W)
  val homeNid     = UInt(p.nodeIdBits.W)
  val opcode      = UInt(4.W)
  val respErr     = UInt(2.W)
  val resp        = UInt(3.W)
  val dataSource  = UInt(8.W)
  val dataPull    = UInt(1.W)
  val cBusy       = UInt(3.W)
  val mecid       = UInt(p.mecidBits.W)
  val dbid        = UInt(16.W)
  val ccid        = UInt(2.W)
  val dataId      = UInt(2.W)
  val cacheLineId = UInt(6.W)
  val tagOp       = UInt(2.W)
  val tag         = UInt(p.tagBits.W)
  val tagUpdate   = UInt(p.tagUpdateBits.W)
  val traceTag    = UInt(1.W)
  val copyAtHome  = UInt(1.W)
  val numDat      = UInt(2.W)
  val replicate   = UInt(1.W)
  val rsvdc       = UInt(p.datRsvdcBits.W)
  val be          = UInt(p.bytesPerBeat.W)
  val data        = UInt(p.dataBits.W)
  val dataCheck   = UInt(p.dataCheckBits.W)
  val poison      = UInt(p.poisonBits.W)
  def fwdState: UInt = dataSource(2, 0)

  def fieldsLSB = Seq(
    qos,
    tgtId,
    srcId,
    txnId,
    homeNid,
    opcode,
    respErr,
    resp,
    dataSource,
    dataPull,
    cBusy,
    mecid,
    dbid,
    ccid,
    dataId,
    cacheLineId,
    tagOp,
    tag,
    tagUpdate,
    traceTag,
    copyAtHome,
    numDat,
    replicate,
    rsvdc,
    be,
    data,
    dataCheck,
    poison
  )

}

class SnoopFlit(p: Params) extends Flit {
  val qos         = UInt(4.W)
  val srcId       = UInt(p.nodeIdBits.W)
  val txnId       = UInt(p.txnIdBits.W)
  val fwdNid      = UInt(p.nodeIdBits.W)
  val fwdTxnId    = UInt(p.txnIdBits.W)
  val opcode      = UInt(5.W)
  val addr        = UInt((p.addressBits - 3).W)
  val pas         = UInt(3.W)
  val doNotGoToSd = UInt(1.W)
  val retToSrc    = UInt(1.W)
  val traceTag    = UInt(1.W)
  val mpam        = UInt(p.mpamBits.W)
  val mecid       = UInt(p.mecidBits.W)

  def fieldsLSB =
    Seq(qos, srcId, txnId, fwdNid, fwdTxnId, opcode, addr, pas, doNotGoToSd, retToSrc, traceTag, mpam, mecid)
}

object Opcode {
  val ReqLCrdReturn         = 0x00
  val ReadShared            = 0x01
  val ReadClean             = 0x02
  val ReadOnce              = 0x03
  val ReadNoSnp             = 0x04
  val PCrdReturn            = 0x05
  val ReadUnique            = 0x07
  val CleanShared           = 0x08
  val CleanInvalid          = 0x09
  val MakeInvalid           = 0x0a
  val CleanUnique           = 0x0b
  val MakeUnique            = 0x0c
  val Evict                 = 0x0d
  val CleanInvalidStorage   = 0x0e
  val ReadNoSnpSep          = 0x11
  val CleanSharedPersistSep = 0x13
  val DVMOp                 = 0x14
  val WriteEvictFull        = 0x15
  val WriteCleanFull        = 0x17
  val WriteUniquePtl        = 0x18
  val WriteUniqueFull       = 0x19
  val WriteBackPtl          = 0x1a
  val WriteBackFull         = 0x1b
  val WriteNoSnpPtl         = 0x1c
  val WriteNoSnpFull        = 0x1d
  val WriteUniqueFullStash  = 0x20
  val WriteUniquePtlStash   = 0x21
  val StashOnceShared       = 0x22
  val StashOnceUnique       = 0x23
  val ReadOnceCleanInvalid  = 0x24
  val ReadOnceMakeInvalid   = 0x25
  val ReadNotSharedDirty    = 0x26
  val CleanSharedPersist    = 0x27
  val AtomicSwap            = 0x38
  val AtomicCompare         = 0x39
  val PrefetchTgt           = 0x3a

  val MakeReadUnique               = 0x41
  val WriteEvictOrEvict            = 0x42
  val WriteUniqueZero              = 0x43
  val WriteNoSnpZero               = 0x44
  val StashOnceSepShared           = 0x47
  val StashOnceSepUnique           = 0x48
  val ReadPreferUnique             = 0x4c
  val CleanInvalidPoPA             = 0x4d
  val WriteNoSnpDef                = 0x4e
  val WriteNoSnpFullCleanSh        = 0x50
  val WriteNoSnpFullCleanInv       = 0x51
  val WriteNoSnpFullCleanShPerSep  = 0x52
  val WriteUniqueFullCleanSh       = 0x54
  val WriteUniqueFullCleanShPerSep = 0x56
  val WriteUniqueFullCleanInvStrg  = 0x57
  val WriteBackFullCleanSh         = 0x58
  val WriteBackFullCleanInv        = 0x59
  val WriteBackFullCleanShPerSep   = 0x5a
  val WriteBackFullCleanInvStrg    = 0x5b
  val WriteCleanFullCleanSh        = 0x5c
  val WriteCleanFullCleanShPerSep  = 0x5e
  val WriteNoSnpPtlCleanSh         = 0x60
  val WriteNoSnpPtlCleanInv        = 0x61
  val WriteNoSnpPtlCleanShPerSep   = 0x62
  val WriteUniquePtlCleanSh        = 0x64
  val WriteUniquePtlCleanShPerSep  = 0x66
  val WriteNoSnpPtlCleanInvPoPA    = 0x70
  val WriteNoSnpFullCleanInvPoPA   = 0x71
  val WriteNoSnpFullCleanInvStrg   = 0x72
  val WriteBackFullCleanInvPoPA    = 0x79

  val AtomicStoreAdd  = 0x28
  val AtomicStoreClr  = 0x29
  val AtomicStoreEor  = 0x2a
  val AtomicStoreSet  = 0x2b
  val AtomicStoreSmax = 0x2c
  val AtomicStoreSmin = 0x2d
  val AtomicStoreUmax = 0x2e
  val AtomicStoreUmin = 0x2f
  val AtomicLoadAdd   = 0x30
  val AtomicLoadClr   = 0x31
  val AtomicLoadEor   = 0x32
  val AtomicLoadSet   = 0x33
  val AtomicLoadSmax  = 0x34
  val AtomicLoadSmin  = 0x35
  val AtomicLoadUmax  = 0x36
  val AtomicLoadUmin  = 0x37

  val RespLCrdReturn = 0x00
  val SnpResp        = 0x01
  val CompAck        = 0x02
  val RetryAck       = 0x03
  val Comp           = 0x04
  val CompDBIDResp   = 0x05
  val DBIDResp       = 0x06
  val PCrdGrant      = 0x07
  val ReadReceipt    = 0x08
  val SnpRespFwded   = 0x09
  val TagMatch       = 0x0a
  val RespSepData    = 0x0b
  val Persist        = 0x0c
  val CompPersist    = 0x0d
  val DBIDRespOrd    = 0x0e
  val StashDone      = 0x10
  val CompStashDone  = 0x11
  val CompCMO        = 0x14

  val SnpLCrdReturn        = 0x00
  val SnpShared            = 0x01
  val SnpClean             = 0x02
  val SnpOnce              = 0x03
  val SnpNotSharedDirty    = 0x04
  val SnpUniqueStash       = 0x05
  val SnpMakeInvalidStash  = 0x06
  val SnpUnique            = 0x07
  val SnpCleanShared       = 0x08
  val SnpCleanInvalid      = 0x09
  val SnpMakeInvalid       = 0x0a
  val SnpStashUnique       = 0x0b
  val SnpStashShared       = 0x0c
  val SnpDVMOp             = 0x0d
  val SnpQuery             = 0x10
  val SnpSharedFwd         = 0x11
  val SnpCleanFwd          = 0x12
  val SnpOnceFwd           = 0x13
  val SnpNotSharedDirtyFwd = 0x14
  val SnpPreferUnique      = 0x15
  val SnpPreferUniqueFwd   = 0x16
  val SnpUniqueFwd         = 0x17

  val DataLCrdReturn              = 0x00
  val SnpRespData                 = 0x01
  val CopyBackWriteData           = 0x02
  val NonCopyBackWriteData        = 0x03
  val CompData                    = 0x04
  val SnpRespDataPtl              = 0x05
  val SnpRespDataFwded            = 0x06
  val WriteDataCancel             = 0x07
  val DataSepResp                 = 0x0b
  val NonCopyBackWriteDataCompAck = 0x0c
}
