package framework.generic.memory.cache

import chisel3.experimental.hierarchy.{Instance, Instantiate}

import framework.generic.queue.Queue

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.generic.memory.cache.configs.CacheParams

@instantiable
class Cache(p: CacheParams, lineQuery: Boolean, pages: Int) extends Module {
  @public
  val io        = IO(new CacheIO(p))
  @public
  val line      = Option.when(lineQuery)(IO(new CacheLine(p)))
  @public
  val rangeLine = Option.when(pages > 0)(IO(new CacheLine(p)))
  @public
  val page      = IO(Vec(pages, new CachePage(p)))

  val valid       = RegInit(VecInit(Seq.fill(p.ways)(VecInit(Seq.fill(p.sets)(false.B)))))
  val tags        = Reg(Vec(p.ways, Vec(p.sets, UInt(p.tagBits.W))))
  val metadata    = Reg(Vec(p.ways, Vec(p.sets, UInt(p.metadataBits.W))))
  val data        = Seq.fill(p.ways)(SyncReadMem(p.sets, Vec(p.lineBytes, UInt(8.W))))
  val nextWay     = RegInit(VecInit(Seq.fill(p.sets)(0.U(p.wayBits.W))))
  val responses   = Instantiate(new Queue(new CacheResponse(p), p.responseDepth, true))
  val pending     = RegInit(false.B)
  val pendingRead = Reg(Bool())
  val pendingWay  = Reg(UInt(p.wayBits.W))
  val result      = Reg(new CacheResponse(p))

  val req     = io.request.bits
  val set     = req.addr(p.offsetBits + p.setBits - 1, p.offsetBits)
  val tag     = req.addr(p.addressBits - 1, p.offsetBits + p.setBits)
  val hits    = VecInit((0 until p.ways).map(w => valid(w)(set) && tags(w)(set) === tag)).asUInt
  val invalid = VecInit((0 until p.ways).map(w => !valid(w)(set) && req.eligible(w))).asUInt

  line.foreach { query =>
    val lineSet  = query.address(p.offsetBits + p.setBits - 1, p.offsetBits)
    val lineTag  = query.address(p.addressBits - 1, p.offsetBits + p.setBits)
    val lineHits = VecInit((0 until p.ways).map(w => valid(w)(lineSet) && tags(w)(lineSet) === lineTag))
    query.hit   := lineHits.asUInt.orR
    query.way   := PriorityEncoder(lineHits.asUInt)
    query.dirty := VecInit((0 until p.ways).map(w => metadata(w)(lineSet)(0)))(query.way)
  }
  rangeLine.foreach { query =>
    val lineSet  = query.address(p.offsetBits + p.setBits - 1, p.offsetBits)
    val lineTag  = query.address(p.addressBits - 1, p.offsetBits + p.setBits)
    val lineHits = VecInit((0 until p.ways).map(w => valid(w)(lineSet) && tags(w)(lineSet) === lineTag))
    query.hit   := lineHits.asUInt.orR
    query.way   := PriorityEncoder(lineHits.asUInt)
    query.dirty := VecInit((0 until p.ways).map(w => metadata(w)(lineSet)(0)))(query.way)
  }

  // The resident lines of each page bucket (page number modulo the sets).
  require(p.addressBits >= 12 + p.setBits)
  def bucket(address: UInt): UInt = address(12 + p.setBits - 1, 12)
  val resident = RegInit(VecInit(Seq.fill(p.sets)(0.U(log2Ceil(p.sets * p.ways + 1).W))))
  page.foreach(query => query.resident := resident(bucket(query.address)) =/= 0.U)

  val order    = VecInit((0 until p.ways).map { offset =>
    val way = if (p.ways == 1) 0.U else (nextWay(set) + offset.U)(p.wayBits - 1, 0)
    req.eligible(way)
  })

  val replacement =
    if (p.ways == 1) 0.U
    else
      (nextWay(set) + PriorityEncoder(order.asUInt))(p.wayBits - 1, 0)

  val lookupWay  = Mux(hits.orR, PriorityEncoder(hits), Mux(invalid.orR, PriorityEncoder(invalid), replacement))
  val lookup     = req.op === CacheOp.Lookup.U
  val selected   = Mux(lookup, lookupWay, req.way)
  val available  = !lookup || hits.orR || req.eligible.orR
  val entryValid = available && valid(selected)(set)
  val read       = (lookup || req.op === CacheOp.Read.U) && entryValid

  io.request.ready       := (responses.io.count +& pending.asUInt) < p.responseDepth.U || responses.io.deq.fire
  responses.io.enq.valid := pending
  responses.io.enq.bits  := result
  io.response <> responses.io.deq
  pending                := io.request.fire
  assert(!pending || responses.io.enq.ready, "Cache response capacity reservation failed")

  val readData = Wire(Vec(p.ways, UInt(p.lineBits.W)))
  for (way <- 0 until p.ways) {
    val write = req.op === CacheOp.Write.U || req.op === CacheOp.Fill.U
    val mask  = Mux(req.op === CacheOp.Fill.U, Fill(p.lineBytes, 1.U(1.W)), req.mask)
    readData(way) := data(way).readWrite(
      set,
      req.data.asTypeOf(Vec(p.lineBytes, UInt(8.W))),
      mask.asBools,
      io.request.fire && selected === way.U && (read || write),
      write
    ).asUInt
  }
  when(pendingRead) {
    responses.io.enq.bits.data := readData(pendingWay)
  }

  when(io.request.fire) {
    assert(req.op <= CacheOp.Invalidate.U, "Unknown Cache operation")
    assert(req.addr(p.offsetBits - 1, 0) === 0.U, "Cache operations require a line-aligned address")
    assert(lookup || req.way < p.ways.U, "Cache way exceeds associativity")
    assert(PopCount(hits) <= 1.U, "Duplicate Cache tag")
    pendingRead       := read
    pendingWay        := selected
    result.id         := req.id
    result.hit        := lookup && hits.orR
    result.available  := available
    result.way        := Mux(available, selected, 0.U)
    result.entryValid := entryValid
    result.addr       := Mux(entryValid, Cat(tags(selected)(set), set, 0.U(p.offsetBits.W)), 0.U)
    result.data       := 0.U
    result.metadata   := Mux(entryValid, metadata(selected)(set), 0.U)

    // A fill adds its line and drops the one it replaces; an invalidation drops its line.
    val replaced = (req.op === CacheOp.Fill.U || req.op === CacheOp.Invalidate.U) && valid(req.way)(set)
    val added    = req.op === CacheOp.Fill.U
    val oldPage  = bucket(Cat(tags(req.way)(set), set, 0.U(p.offsetBits.W)))
    val newPage  = bucket(req.addr)
    when(!(replaced && added && oldPage === newPage)) {
      when(added)(resident(newPage)    := resident(newPage) + 1.U)
      when(replaced)(resident(oldPage) := resident(oldPage) - 1.U)
    }
    when(req.op === CacheOp.Fill.U) {
      assert(!hits.orR || hits(req.way), "Cache fill would create a duplicate tag")
      valid(req.way)(set)    := true.B
      tags(req.way)(set)     := tag
      metadata(req.way)(set) := req.metadata
      nextWay(set)           := (if (p.ways == 1) 0.U else (req.way + 1.U)(p.wayBits - 1, 0))
      result.entryValid      := true.B
      result.addr            := req.addr
      result.metadata        := req.metadata
    }
    when(req.op === CacheOp.Write.U) {
      assert(entryValid && hits(req.way), "Cache write requires the matching resident line")
      metadata(req.way)(set) := req.metadata
      result.metadata        := req.metadata
    }
    when(req.op === CacheOp.Invalidate.U) {
      valid(req.way)(set) := false.B
      result.entryValid   := false.B
      result.addr         := 0.U
      result.metadata     := 0.U
    }
  }
}
