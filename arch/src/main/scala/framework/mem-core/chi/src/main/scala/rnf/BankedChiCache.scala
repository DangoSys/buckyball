package memcore.bus.chi.rnf

import chisel3._
import chisel3.util._
import memcore.bus.chi._

// Each bank executes one demand operation while serving snoops independently.
// The retirement FIFO preserves the order of the untagged CPU result interface.
class BankedChiCache(
  p:          Params,
  nodeId:     Int,
  cacheLines: Int = 4,
  homeId:     Int = 64,
  homeCount:  Int = 1,
  banks:      Int = 2)
    extends Module {
  require(banks >= 1 && banks <= 256 && isPow2(banks))
  require(cacheLines >= 2 * banks && isPow2(cacheLines))
  private val bankBits     = math.max(1, log2Ceil(banks))
  private val linesPerBank = cacheLines / banks

  val io = IO(new Bundle {
    val access      = Flipped(Decoupled(new CacheAccess(p)))
    val result      = Decoupled(new CacheResult)
    val chi         = new RequesterPort(p)
    val hits        = Output(UInt(32.W))
    val misses      = Output(UInt(32.W))
    val directory   = Output(Vec(cacheLines, new CacheLineState(p)))
    val outstanding = Output(UInt(32.W))
  })

  val caches = Seq.tabulate(banks) { i =>
    Module(new ChiCache(p, nodeId, linesPerBank, homeId, homeCount, txnId = i, bankCount = banks))
  }

  def bankOf(address: UInt): UInt =
    if (banks == 1) 0.U(0.W) else address(log2Ceil(banks) + 5, 6)
  val selected      = bankOf(io.access.bits.addr)
  val retirement    = Module(new Queue(UInt(bankBits.W), banks, pipe = true))
  val serial        = io.access.bits.atomic =/= CacheAtomic.None.U
  val serialActive  = RegInit(false.B)
  val serialSC      = RegInit(false.B)
  val canAccept     = !serialActive && (!serial || !retirement.io.deq.valid)
  val selectedReady = VecInit(caches.map(_.io.access.ready))(selected)
  io.access.ready         := canAccept && retirement.io.enq.ready && selectedReady
  retirement.io.enq.valid := io.access.valid && canAccept && selectedReady
  retirement.io.enq.bits  := selected
  for (i <- 0 until banks) {
    caches(i).io.access.valid := io.access.valid && canAccept && retirement.io.enq.ready && selected === i.U
    caches(i).io.access.bits  := io.access.bits
  }
  when(io.access.fire && serial) {
    serialActive := true.B
    serialSC     := io.access.bits.atomic === CacheAtomic.SC.U
  }

  val oldest = if (banks == 1) 0.U(0.W) else retirement.io.deq.bits
  io.result.valid         := retirement.io.deq.valid && VecInit(caches.map(_.io.result.valid))(oldest)
  io.result.bits          := VecInit(caches.map(_.io.result.bits))(oldest)
  retirement.io.deq.ready := io.result.fire
  for (i <- 0 until banks) {
    caches(i).io.result.ready := retirement.io.deq.valid && oldest === i.U && io.result.ready
  }
  when(io.result.fire && serialActive)(serialActive := false.B)
  io.outstanding := retirement.io.count

  // A hart has one LR reservation across all banks. Any SC consumes it, even
  // when the SC addresses another bank and fails there.
  val newLR       = io.access.fire && io.access.bits.atomic === CacheAtomic.LR.U
  val completedSC = io.result.fire && serialActive && serialSC
  for (cache <- caches) { cache.io.dropReservation := newLR || completedSC }

  val reqArb   = Module(new RRArbiter(new RequestFlit(p), banks))
  val rspArb   = Module(new RRArbiter(new ResponseFlit(p), banks))
  val datArb   = Module(new RRArbiter(new DataFlit(p), banks))
  val reqQueue = Module(new Queue(new RequestFlit(p), 2, pipe = true))
  val rspQueue = Module(new Queue(new ResponseFlit(p), 2, pipe = true))
  val datQueue = Module(new Queue(new DataFlit(p), 2, pipe = true))
  for (i <- 0 until banks) {
    reqArb.io.in(i) <> caches(i).io.chi.req
    rspArb.io.in(i) <> caches(i).io.chi.txRsp
    datArb.io.in(i) <> caches(i).io.chi.txDat
  }
  reqQueue.io.enq <> reqArb.io.out
  rspQueue.io.enq <> rspArb.io.out
  datQueue.io.enq <> datArb.io.out
  io.chi.req <> reqQueue.io.deq
  io.chi.txRsp <> rspQueue.io.deq
  io.chi.txDat <> datQueue.io.deq

  val rspBank = if (banks == 1) 0.U(0.W) else io.chi.rxRsp.bits.txnId(bankBits - 1, 0)
  val datBank = if (banks == 1) 0.U(0.W) else io.chi.rxDat.bits.txnId(bankBits - 1, 0)
  val snpBank = bankOf(io.chi.snp.bits.addr << 3)
  io.chi.rxRsp.ready := VecInit(caches.map(_.io.chi.rxRsp.ready))(rspBank)
  io.chi.rxDat.ready := VecInit(caches.map(_.io.chi.rxDat.ready))(datBank)
  io.chi.snp.ready   := VecInit(caches.map(_.io.chi.snp.ready))(snpBank)
  when(io.chi.rxRsp.valid)(assert(io.chi.rxRsp.bits.txnId < banks.U, "Unknown CPU RSP transaction bank"))
  when(io.chi.rxDat.valid)(assert(io.chi.rxDat.bits.txnId < banks.U, "Unknown CPU DAT transaction bank"))
  for (i <- 0 until banks) {
    val chi = caches(i).io.chi
    chi.rxRsp.valid := io.chi.rxRsp.valid && rspBank === i.U
    chi.rxRsp.bits  := io.chi.rxRsp.bits
    chi.rxDat.valid := io.chi.rxDat.valid && datBank === i.U
    chi.rxDat.bits  := io.chi.rxDat.bits
    chi.snp.valid   := io.chi.snp.valid && snpBank === i.U
    chi.snp.bits    := io.chi.snp.bits
    for (line <- 0 until linesPerBank) {
      io.directory(line * banks + i) := caches(i).io.directory(line)
    }
  }
  io.hits := caches.map(_.io.hits).reduce(_ + _)
  io.misses := caches.map(_.io.misses).reduce(_ + _)
}

object EmitBankedChiCache extends App {
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new BankedChiCache(Params(), nodeId = 1),
    firtoolOpts = args.drop(1) ++ Seq("--split-verilog", "-o=build"),
    args = Array("--target-dir", "build")
  )
}
