package examples.core.xiangshan

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import framework.ip.bus.chi

/** Credit-accounted native CHI B link and explicit framework flit translation. */
@instantiable
class Chi(p: chi.Params, homeId: Int) extends Module {
  require(p.nodeIdBits == 7 && p.addressBits == 48 && p.dataBits == 256)

  @public
  val io = IO(new Bundle {
    val native    = Flipped(new Native.Link)
    val requester = new chi.RequesterPort(p)
  })

  io.native.tx.linkactiveack := RegNext(io.native.tx.linkactivereq, false.B)
  io.native.rx.linkactivereq := !reset.asBool
  io.native.rxsactive        := !reset.asBool
  io.native.syscoack         := RegNext(io.native.syscoreq, false.B)
  val receiving = io.native.tx.linkactivereq && io.native.tx.linkactiveack
  val sending   = io.native.rx.linkactivereq && io.native.rx.linkactiveack

  val requestRx = Instantiate(new chi.Rx((new Native.Request).getWidth, 4))
  requestRx.io.active := receiving
  requestRx.io.link <> io.native.tx.req
  val request = Wire(new Native.Request)
  request.unpack(requestRx.io.out.bits)
  io.requester.req.valid                    := requestRx.io.out.valid
  requestRx.io.out.ready                    := io.requester.req.ready
  io.requester.req.bits                     := 0.U.asTypeOf(io.requester.req.bits)
  io.requester.req.bits.qos                 := request.qos
  io.requester.req.bits.srcId               := request.srcId
  io.requester.req.bits.txnId               := request.txnId
  io.requester.req.bits.returnNid           := request.returnNid
  io.requester.req.bits.stashNidValidEndian := request.stashNidValidEndian
  io.requester.req.bits.returnTxnId         := request.returnTxnId
  io.requester.req.bits.opcode              := request.opcode
  io.requester.req.bits.size                := request.size
  io.requester.req.bits.addr                := request.addr
  io.requester.req.bits.likelyShared        := request.likelyShared
  io.requester.req.bits.allowRetry          := request.allowRetry
  io.requester.req.bits.order               := request.order
  io.requester.req.bits.pCrdType            := request.pCrdType
  io.requester.req.bits.memAttr             := request.memAttr
  io.requester.req.bits.snpAttr             := request.snpAttr
  io.requester.req.bits.lpid                := request.lpid
  io.requester.req.bits.exclSnoopMe         := request.exclSnoopMe
  io.requester.req.bits.expCompAck          := request.expCompAck
  io.requester.req.bits.traceTag            := request.traceTag
  io.requester.req.bits.tgtId               := homeId.U
  io.requester.req.bits.pas                 := request.ns.asUInt
  when(requestRx.io.out.valid) {
    assert(request.tgtId === 0.U || request.tgtId === homeId.U)
    assert(request.rsvdc === 0.U)
  }

  val responseRx = Instantiate(new chi.Rx((new Native.Response).getWidth, 4))
  responseRx.io.active := receiving
  responseRx.io.link <> io.native.tx.rsp
  val response = Wire(new Native.Response)
  response.unpack(responseRx.io.out.bits)
  io.requester.txRsp.valid         := responseRx.io.out.valid
  responseRx.io.out.ready          := io.requester.txRsp.ready
  io.requester.txRsp.bits          := 0.U.asTypeOf(io.requester.txRsp.bits)
  io.requester.txRsp.bits.qos      := response.qos
  io.requester.txRsp.bits.tgtId    := response.tgtId
  io.requester.txRsp.bits.srcId    := response.srcId
  io.requester.txRsp.bits.txnId    := response.txnId
  io.requester.txRsp.bits.opcode   := response.opcode
  io.requester.txRsp.bits.respErr  := response.respErr
  io.requester.txRsp.bits.resp     := response.resp
  io.requester.txRsp.bits.fwdState := response.fwdState
  io.requester.txRsp.bits.dbid     := response.dbid
  io.requester.txRsp.bits.pCrdType := response.pCrdType
  io.requester.txRsp.bits.traceTag := response.traceTag

  val dataRx = Instantiate(new chi.Rx((new Native.Data).getWidth, 4))
  dataRx.io.active := receiving
  dataRx.io.link <> io.native.tx.dat
  val data = Wire(new Native.Data)
  data.unpack(dataRx.io.out.bits)
  io.requester.txDat.valid           := dataRx.io.out.valid
  dataRx.io.out.ready                := io.requester.txDat.ready
  io.requester.txDat.bits            := 0.U.asTypeOf(io.requester.txDat.bits)
  io.requester.txDat.bits.qos        := data.qos
  io.requester.txDat.bits.tgtId      := data.tgtId
  io.requester.txDat.bits.srcId      := data.srcId
  io.requester.txDat.bits.txnId      := data.txnId
  io.requester.txDat.bits.homeNid    := data.homeNid
  io.requester.txDat.bits.opcode     := data.opcode
  io.requester.txDat.bits.respErr    := data.respErr
  io.requester.txDat.bits.resp       := data.resp
  io.requester.txDat.bits.dataSource := data.dataSource
  io.requester.txDat.bits.dbid       := data.dbid
  io.requester.txDat.bits.ccid       := data.ccid
  io.requester.txDat.bits.dataId     := data.dataId
  io.requester.txDat.bits.traceTag   := data.traceTag
  io.requester.txDat.bits.be         := data.be
  io.requester.txDat.bits.data       := data.data
  when(dataRx.io.out.valid)(assert(data.rsvdc === 0.U))

  val incomingResponseTx = Instantiate(new chi.Tx((new Native.Response).getWidth, 15))
  incomingResponseTx.io.active := sending
  io.native.rx.rsp <> incomingResponseTx.io.link
  val incomingResponse = Wire(new Native.Response)
  incomingResponse               := 0.U.asTypeOf(incomingResponse)
  incomingResponse.qos           := io.requester.rxRsp.bits.qos
  incomingResponse.tgtId         := io.requester.rxRsp.bits.tgtId
  incomingResponse.srcId         := io.requester.rxRsp.bits.srcId
  incomingResponse.txnId         := io.requester.rxRsp.bits.txnId(7, 0)
  incomingResponse.opcode        := io.requester.rxRsp.bits.opcode(3, 0)
  incomingResponse.respErr       := io.requester.rxRsp.bits.respErr
  incomingResponse.resp          := io.requester.rxRsp.bits.resp
  incomingResponse.fwdState      := io.requester.rxRsp.bits.fwdState
  incomingResponse.dbid          := io.requester.rxRsp.bits.dbid(7, 0)
  incomingResponse.pCrdType      := io.requester.rxRsp.bits.pCrdType
  incomingResponse.traceTag      := io.requester.rxRsp.bits.traceTag.asBool
  incomingResponseTx.io.in.valid := io.requester.rxRsp.valid
  incomingResponseTx.io.in.bits  := incomingResponse.packed
  io.requester.rxRsp.ready       := incomingResponseTx.io.in.ready
  when(io.requester.rxRsp.valid) {
    assert(io.requester.rxRsp.bits.txnId < 256.U)
    assert(io.requester.rxRsp.bits.dbid < 256.U && io.requester.rxRsp.bits.opcode < 16.U)
    assert(
      io.requester.rxRsp.bits.cBusy === 0.U && io.requester.rxRsp.bits.tagOp === 0.U && io.requester.rxRsp.bits.cacheLineId === 0.U
    )
  }

  val incomingDataTx = Instantiate(new chi.Tx((new Native.Data).getWidth, 15))
  incomingDataTx.io.active := sending
  io.native.rx.dat <> incomingDataTx.io.link
  val incomingData = Wire(new Native.Data)
  incomingData               := 0.U.asTypeOf(incomingData)
  incomingData.qos           := io.requester.rxDat.bits.qos
  incomingData.tgtId         := io.requester.rxDat.bits.tgtId
  incomingData.srcId         := io.requester.rxDat.bits.srcId
  incomingData.txnId         := io.requester.rxDat.bits.txnId(7, 0)
  incomingData.homeNid       := io.requester.rxDat.bits.homeNid
  incomingData.opcode        := io.requester.rxDat.bits.opcode(2, 0)
  incomingData.respErr       := io.requester.rxDat.bits.respErr
  incomingData.resp          := io.requester.rxDat.bits.resp
  incomingData.dataSource    := io.requester.rxDat.bits.dataSource(2, 0)
  incomingData.dbid          := io.requester.rxDat.bits.dbid(7, 0)
  incomingData.ccid          := io.requester.rxDat.bits.ccid
  incomingData.dataId        := io.requester.rxDat.bits.dataId
  incomingData.traceTag      := io.requester.rxDat.bits.traceTag.asBool
  incomingData.be            := io.requester.rxDat.bits.be
  incomingData.data          := io.requester.rxDat.bits.data
  incomingDataTx.io.in.valid := io.requester.rxDat.valid
  incomingDataTx.io.in.bits  := incomingData.packed
  io.requester.rxDat.ready   := incomingDataTx.io.in.ready
  when(io.requester.rxDat.valid) {
    assert(io.requester.rxDat.bits.txnId < 256.U)
    assert(
      io.requester.rxDat.bits.dbid < 256.U && io.requester.rxDat.bits.opcode < 8.U && io.requester.rxDat.bits.dataSource < 8.U
    )
    assert(io.requester.rxDat.bits.cBusy === 0.U && io.requester.rxDat.bits.tagOp === 0.U)
    assert(io.requester.rxDat.bits.tag === 0.U && io.requester.rxDat.bits.tagUpdate === 0.U)
    assert(
      io.requester.rxDat.bits.copyAtHome === 0.U && io.requester.rxDat.bits.numDat === 0.U && io.requester.rxDat.bits.replicate === 0.U
    )
    assert(io.requester.rxDat.bits.cacheLineId === 0.U && io.requester.rxDat.bits.dataPull === 0.U)
  }

  val snoopTx = Instantiate(new chi.Tx((new Native.Snoop).getWidth, 4))
  snoopTx.io.active := sending
  io.native.rx.snp <> snoopTx.io.link
  val snoop = Wire(new Native.Snoop)
  snoop                  := 0.U.asTypeOf(snoop)
  snoop.qos              := io.requester.snp.bits.qos
  snoop.srcId            := io.requester.snp.bits.srcId
  snoop.txnId            := io.requester.snp.bits.txnId(7, 0)
  snoop.fwdNid           := io.requester.snp.bits.fwdNid
  snoop.fwdTxnId         := io.requester.snp.bits.fwdTxnId(7, 0)
  snoop.opcode           := io.requester.snp.bits.opcode(4, 0)
  snoop.addr             := io.requester.snp.bits.addr
  snoop.doNotGoToSd      := io.requester.snp.bits.doNotGoToSd.asBool
  snoop.retToSrc         := io.requester.snp.bits.retToSrc.asBool
  snoop.traceTag         := io.requester.snp.bits.traceTag.asBool
  snoop.ns               := io.requester.snp.bits.pas(0)
  snoopTx.io.in.valid    := io.requester.snp.valid
  snoopTx.io.in.bits     := snoop.packed
  io.requester.snp.ready := snoopTx.io.in.ready
  when(io.requester.snp.valid) {
    assert(io.requester.snp.bits.txnId < 256.U)
    assert(io.requester.snp.bits.fwdTxnId < 256.U && io.requester.snp.bits.pas < 2.U)
  }

}
