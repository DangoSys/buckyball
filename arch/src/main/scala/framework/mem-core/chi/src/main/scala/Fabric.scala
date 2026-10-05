package memcore.bus.chi

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}

/** Explicit requester-to-Home CHI fabric; no caches or backing-memory ownership. */
@instantiable
class Fabric(p: Params, requesterCount: Int, homeId: Int) extends Module {
  require(requesterCount >= 1 && requesterCount < homeId && BigInt(homeId) < (BigInt(1) << p.nodeIdBits))
  private val c = p

  @public
  val io = IO(new Bundle {
    val active            = Input(Bool())
    val blockRequesterRsp = Input(Bool())
    val requesters        = Vec(requesterCount, Flipped(new RequesterPort(c)))
    val req               = Decoupled(new RequestFlit(c))
    val rxRsp             = Decoupled(new ResponseFlit(c))
    val rxDat             = Decoupled(new DataFlit(c))
    val rsp               = Flipped(Decoupled(new ResponseFlit(c)))
    val dat               = Flipped(Decoupled(new DataFlit(c)))
    val snp               = Flipped(Decoupled(new DirectedSnoop(c)))
  })

  def creditLink[T <: Flit](source: DecoupledIO[T], sink: DecoupledIO[T], blockRx: Bool = false.B): Unit = {
    // Backpressure retains complete flits in the real credit-accounted Rx FIFO.
    // Active and FLITV retain their channel semantics throughout the stall.
    val tx: Instance[Tx] = Instantiate(new Tx(source.bits.flitWidth, maxCredits = 4))
    val rx: Instance[Rx] = Instantiate(new Rx(source.bits.flitWidth, depth = 4))
    tx.io.active    := io.active
    rx.io.active    := io.active
    tx.io.in.valid  := source.valid
    tx.io.in.bits   := source.bits.packed
    source.ready    := tx.io.in.ready
    rx.io.link <> tx.io.link
    sink.valid      := rx.io.out.valid && !blockRx
    sink.bits.unpack(rx.io.out.bits)
    rx.io.out.ready := sink.ready && !blockRx
  }

  // All three arbiters use initialized history: the default RR lastGrant is not
  // reset, so simultaneous first requests can otherwise select unknown data.
  def requesterArbiter[T <: Data](flit: T): RRArbiter[T] = Module(new RRArbiter(flit, requesterCount) {

    override lazy val lastGrant = {
      val cursor = RegInit(0.U(math.max(1, log2Ceil(requesterCount)).W))
      when(io.out.fire)(cursor := io.chosen)
      cursor
    }

  })

  val requests  = requesterArbiter(new RequestFlit(c))
  val responses = requesterArbiter(new ResponseFlit(c))
  val data      = requesterArbiter(new DataFlit(c))
  io.req <> requests.io.out
  io.rxRsp <> responses.io.out
  io.rxDat <> data.io.out
  for (i <- 0 until requesterCount) {
    creditLink(io.requesters(i).req, requests.io.in(i))
    creditLink(io.requesters(i).txRsp, responses.io.in(i), io.blockRequesterRsp)
    creditLink(io.requesters(i).txDat, data.io.in(i))
    when(requests.io.in(i).valid) {
      assert(
        requests.io.in(i).bits.srcId === (i + 1).U && requests.io.in(i).bits.tgtId === homeId.U,
        "CacheSystem request crossed a wrong node or role"
      )
    }
    when(responses.io.in(i).valid)(assert(
      responses.io.in(i).bits.srcId === (i + 1).U &&
        responses.io.in(i).bits.tgtId === homeId.U,
      "CacheSystem response crossed a wrong node or role"
    ))
    when(data.io.in(i).valid)(assert(
      data.io.in(i).bits.srcId === (i + 1).U &&
        data.io.in(i).bits.tgtId === homeId.U,
      "CacheSystem data crossed a wrong node or role"
    ))
  }
  val rsp = Wire(Vec(requesterCount, Decoupled(new ResponseFlit(c))))
  val dat = Wire(Vec(requesterCount, Decoupled(new DataFlit(c))))
  val snp = Wire(Vec(requesterCount, Decoupled(new SnoopFlit(c))))
  for (i <- 0 until requesterCount) {
    rsp(i).valid := io.rsp.valid && io.rsp.bits.tgtId === (i + 1).U
    rsp(i).bits  := io.rsp.bits
    dat(i).valid := io.dat.valid && io.dat.bits.tgtId === (i + 1).U
    dat(i).bits  := io.dat.bits
    snp(i).valid := io.snp.valid && io.snp.bits.targetNode === (i + 1).U
    snp(i).bits  := io.snp.bits.flit
    creditLink(rsp(i), io.requesters(i).rxRsp)
    creditLink(dat(i), io.requesters(i).rxDat)
    creditLink(snp(i), io.requesters(i).snp)
  }
  io.rsp.ready := (0 until requesterCount).map(i => rsp(i).ready && io.rsp.bits.tgtId === (i + 1).U).reduce(_ || _)
  io.dat.ready := (0 until requesterCount).map(i => dat(i).ready && io.dat.bits.tgtId === (i + 1).U).reduce(_ || _)
  io.snp.ready := (0 until requesterCount).map(i => snp(i).ready && io.snp.bits.targetNode === (i + 1).U).reduce(_ || _)
  when(io.rsp.valid)(assert(
    io.rsp.bits.tgtId >= 1.U && io.rsp.bits.tgtId <= requesterCount.U,
    "CacheSystem response target is not a requester"
  ))
  when(io.dat.valid)(assert(
    io.dat.bits.tgtId >= 1.U && io.dat.bits.tgtId <= requesterCount.U,
    "CacheSystem data target is not a requester"
  ))
  when(io.snp.valid)(assert(
    io.snp.bits.targetNode >= 1.U && io.snp.bits.targetNode <= requesterCount.U,
    "CacheSystem snoop target is not a requester"
  ))
}
