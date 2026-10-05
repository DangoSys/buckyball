package hier.tile.memory

import memcore.memory.queue.Queue

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import hier.core.rocket.{Cache => CoreCache}
import memcore.bus.chi.{DataFlit, DirectedSnoop, RequestFlit}
import memcore.bus.chi.rnf.{BankedChiCache, CacheAccess, CacheAtomic, CacheResult, RnfParams}
import memcore.bus.chi.snf.{LineRequest, LineResponse}
import memcore.memory.coherence.configs.CoherenceParams
import memcore.memory.interlock.{AccessInfo, Acknowledgement, Dispatch, Interlock, Params, Tag}

/** Physical-access verification composition: actual L1s, one shared L2 and one core-local interlock. */
@instantiable
class Consistency(memory: CoherenceParams, l1: RnfParams, tracking: Params) extends Module {
  require(memory.agents >= 2 && l1.homeCount == 1 && l1.homeId == memory.homeId)
  require(l1.chi == memory.chi && tracking.addressBits == memory.chi.addressBits)
  private val c = memory.chi

  @public
  val io = IO(new Bundle {
    val access               = Flipped(Vec(memory.agents, Decoupled(new CacheAccess(c))))
    val result               = Vec(memory.agents, Decoupled(new CacheResult))
    val dispatch             = Flipped(Decoupled(new Dispatch(tracking)))
    val accessInfo           = Flipped(Decoupled(new AccessInfo(tracking)))
    val grant                = Decoupled(new Tag(tracking))
    val done                 = Flipped(Decoupled(new Acknowledgement(tracking)))
    val complete             = Decoupled(new Tag(tracking))
    val olderDispatchPending = Input(Bool())
    val olderRequestsDrained = Input(Bool())
    val blockRequesterRsp    = Input(Bool())
    val blockRequesterData   = Input(Bool())
    val cpuAllow             = Output(Bool())
    val memoryReq            = Decoupled(new LineRequest(c))
    val memoryResp           = Flipped(Decoupled(new LineResponse(c)))
    val observedReq          = Output(Valid(new RequestFlit(c)))
    val observedSnp          = Output(Valid(new DirectedSnoop(c)))
    val outstanding          = Output(UInt(log2Ceil(memory.mshrEntries + 1).W))
  })

  val home: Instance[Home] = Instantiate(new Home(memory))
  home.io.active            := true.B
  home.io.blockRequesterRsp := io.blockRequesterRsp
  io.memoryReq <> home.io.memoryReq
  home.io.memoryResp <> io.memoryResp
  io.observedReq            := home.io.observedReq
  io.observedSnp            := home.io.observedSnp
  io.outstanding            := home.io.outstanding

  val interlock: Instance[Interlock] = Instantiate(new Interlock(tracking))
  interlock.io.cancel.valid := false.B
  interlock.io.cancel.bits  := 0.U.asTypeOf(new Dispatch(tracking))
  interlock.io.dispatch <> io.dispatch
  interlock.io.accessInfo <> io.accessInfo
  io.grant <> interlock.io.grant
  interlock.io.done <> io.done
  io.complete <> interlock.io.complete
  val access = io.access(0).bits
  interlock.io.cpuQuery.valid                := io.access(0).valid && access.atomic =/= CacheAtomic.Fence.U
  interlock.io.cpuQuery.paddr                := access.addr
  interlock.io.cpuQuery.sizeLog2             := Mux(access.atomicWord, 2.U, 3.U)
  interlock.io.cpuQuery.write                := access.write || access.atomic === CacheAtomic.SC.U ||
    (access.atomic >= CacheAtomic.Swap.U && access.atomic <= CacheAtomic.MaxU.U)
  interlock.io.cpuQuery.olderDispatchPending := io.olderDispatchPending
  io.cpuAllow                                := interlock.io.cpuAllow

  val first: Instance[CoreCache] = Instantiate(new CoreCache(l1.copy(nodeId = 1), tracking))
  first.io.access.valid         := io.access(0).valid && interlock.io.cpuAllow
  first.io.access.bits          := access
  io.access(0).ready            := first.io.access.ready && interlock.io.cpuAllow
  io.result(0) <> first.io.result
  first.io.olderRequestsDrained := io.olderRequestsDrained
  first.io.maintenance <> interlock.io.maintenance
  interlock.io.maintained <> first.io.maintained
  home.io.requesters(0) <> first.io.chi
  // Verification-only delayed transport, including a complete old writeback
  // accepted by L1 but not yet delivered to Home.
  val delayedData = Module(new Queue(new DataFlit(c), c.beatsPerLine))
  delayedData.io.enq <> first.io.chi.txDat
  home.io.requesters(0).txDat.valid := delayedData.io.deq.valid && !io.blockRequesterData
  home.io.requesters(0).txDat.bits  := delayedData.io.deq.bits
  delayedData.io.deq.ready          := home.io.requesters(0).txDat.ready && !io.blockRequesterData
  for (i <- 1 until memory.agents) {
    val cache: Instance[BankedChiCache] = Instantiate(new BankedChiCache(l1.copy(nodeId = i + 1)))
    cache.io.access <> io.access(i)
    io.result(i) <> cache.io.result
    home.io.requesters(i) <> cache.io.chi
  }
}
