package framework.generic.gvn

import chisel3._
import chisel3.util._
import framework.generic.bus.axi4

/**
 * Config port word write. Word address map:
 *   - 0x0000 + 4 * entry + word: TMT entry words (see CheckerOnChip)
 *   - 0x1000 + bucket: reference bucket V_j (signed, low accBits bits)
 *   - 0x2000: sign seed; 0x2001: start (clears entry counter, hash and fault)
 */
class ConfigWrite extends Bundle {
  val addr = UInt(14.W)
  val data = UInt(64.W)
}

class Verdict(p: Params) extends Bundle {
  val entry        = UInt(p.entryBits.W)
  val hashFail     = Bool()
  val rangeFail    = Bool()
  val kurtosisFail = Bool()
  val hash         = UInt(32.W)
  val max          = SInt(p.devBits.W)
  val min          = SInt(p.devBits.W)
  val s2           = UInt(p.s2Bits.W)
  val s4           = UInt(p.s4Bits.W)
}

/**
 * Checker on Chip (CoC): one per core, snoops the core's NPU DMA write channel to DDR.
 *
 * HC: hash += rowHash(beat address, strobe-masked data) for every write beat of a subgraph outside the SC window;
 * compared with the TMT entry's expected hash at the subgraph's fence.
 *
 * SC: beats inside the entry's checked tensor window add sum_i(s_i * y_i / Delta) over their lanes (INT8 x16, INT32
 * x4, FP32 x4 with Delta = 2^fpExp) into bucket (beat address mod B). Buckets are preloaded with -V_j (the GPU
 * reference in bucket form), so at the fence bucket j holds the deviation d_j. The fence streams the B buckets
 * through range = max - min and kurtosis B * sum(d^4) / sum(d^2)^2 and compares them with T_r and T_k.
 *
 * TMT entry words: w0 = expected hash; w1 = window base (bytes, 16-byte aligned); w2 = window bytes [31:0] |
 * format [33:32] (0 INT8, 1 INT32, 2 FP32) | fpExp [47:40] (signed) | scEnable [48]; w3 = T_r [31:0] | T_k [47:32]
 * (unsigned 8.8).
 */
class CheckerOnChip(p: Params) extends Module {

  val io = IO(new Bundle {
    val aw         = Flipped(Valid(new axi4.Address(p.axi)))
    val w          = Flipped(Valid(new axi4.WriteData(p.axi)))
    val config     = Flipped(Valid(new ConfigWrite))
    val fence      = Input(Bool())
    val fenceReady = Output(Bool())
    val verdict    = Valid(new Verdict(p))
    val fault      = Output(Bool())
  })

  def rotateLeft(value: UInt, amount: Int): UInt =
    Cat(value(31 - amount, 0), value(31, 32 - amount))

  def xorshift(x: UInt): UInt = {
    val a = x ^ (x << 13)(31, 0)
    val b = a ^ (a >> 17)
    b ^ (b << 5)(31, 0)
  }

  // FP32 -> y / 2^exp, truncated toward zero; saturates on overflow, Inf and NaN; flushes denormals.
  def fpToFixed(bits: UInt, exp: SInt): SInt = {
    val e         = bits(30, 23)
    val mantissa  = Cat(1.U(1.W), bits(22, 0))
    val shift     = e.zext - 150.S - exp
    val big       = shift > 7.S || e === 255.U
    val tiny      = shift < -24.S || e === 0.U
    val left      = (mantissa << shift.asUInt(2, 0))(30, 0)
    val right     = mantissa >> (0.S - shift).asUInt(4, 0)
    val magnitude = Mux(big, "h7fffffff".U(31.W), Mux(tiny, 0.U, Mux(shift >= 0.S, left, right)))
    Mux(bits(31), 0.S - magnitude.zext, magnitude.zext)
  }

  // Snooped AXI channels; W may arrive before its AW, so both are queued and paired in order.
  val awQueue = Module(new Queue(new axi4.Address(p.axi), p.awDepth))
  val wQueue  = Module(new Queue(new axi4.WriteData(p.axi), p.wDepth))
  awQueue.io.enq.valid := io.aw.valid
  awQueue.io.enq.bits  := io.aw.bits
  wQueue.io.enq.valid  := io.w.valid
  wQueue.io.enq.bits   := io.w.bits
  assert(!io.aw.valid || awQueue.io.enq.ready, "CoC AW snoop queue overflow")
  assert(!io.w.valid || wQueue.io.enq.ready, "CoC W snoop queue overflow")

  val head      = awQueue.io.deq.bits
  val beatIndex = RegInit(0.U(8.W))
  val beat      = awQueue.io.deq.valid && wQueue.io.deq.valid
  val lastBeat  = beatIndex === head.len
  wQueue.io.deq.ready  := awQueue.io.deq.valid
  awQueue.io.deq.ready := wQueue.io.deq.valid && lastBeat
  when(beat)(beatIndex := Mux(lastBeat, 0.U, beatIndex + 1.U))
  val beatAddr = head.addr(p.axi.addressBits - 1, 4) + beatIndex
  val data     = wQueue.io.deq.bits.data
  val strb     = wQueue.io.deq.bits.strb

  // TMT and control state.
  val tmt     = Seq.fill(4)(Mem(p.entries, UInt(64.W)))
  val seed    = RegInit(0.U(32.W))
  val current = RegInit(0.U(p.entryBits.W))
  val hash    = RegInit(0.U(32.W))
  val fault   = RegInit(false.B)
  val region  = io.config.bits.addr(13, 12)
  val offset  = io.config.bits.addr(11, 0)
  for (word <- 0 until 4)
    when(io.config.valid && region === 0.U && offset(1, 0) === word.U) {
      tmt(word).write(offset(11, 2)(p.entryBits - 1, 0), io.config.bits.data)
    }
  val start = io.config.valid && region === 2.U && offset === 1.U
  when(io.config.valid && region === 2.U && offset === 0.U)(seed := io.config.bits.data(31, 0))

  val expectedHash = tmt(0).read(current)(31, 0)
  val windowBase   = tmt(1).read(current)(p.axi.addressBits - 1, 4)
  val control      = tmt(2).read(current)
  val limits       = tmt(3).read(current)
  val windowBeats  = control(31, 4)
  val format       = control(33, 32)
  val fpExp        = control(47, 40).asSInt
  val scEnable     = control(48)
  val rangeLimit   = limits(31, 0)
  val kurtLimit    = limits(47, 32)

  // HC: order-independent additive hash, the BankHash row mix applied to each beat outside the SC window (the window
  // is value-checked, not bit-exact).
  val inWindow   = scEnable && beatAddr >= windowBase && beatAddr < windowBase + windowBeats
  val row        = data & FillInterleaved(8, strb)
  val addressKey = beatAddr(31, 0) ^ beatAddr(p.axi.addressBits - 5, 32).pad(32)
  val rowHash    = row(31, 0) ^ rotateLeft(row(63, 32), 7) ^ rotateLeft(row(95, 64), 13) ^
    rotateLeft(row(127, 96), 21) ^ rotateLeft(addressKey, 11)
  when(start)(hash := 0.U).elsewhen(beat && !inWindow)(hash := hash + rowHash)

  // SC lanes: one signed sum per beat.
  val signs      = xorshift(xorshift(beatAddr(31, 0) ^ seed))

  val byteLanes = (0 until 16).map { i =>
    val value = Mux(format === 0.U && strb(i), data(8 * i + 7, 8 * i).asSInt, 0.S).pad(34)
    Mux(signs(i), 0.S - value, value)
  }

  val wordLanes = (0 until 4).map { i =>
    val bits   = data(32 * i + 31, 32 * i)
    val chosen = Mux(format === 1.U, bits.asSInt, fpToFixed(bits, fpExp))
    val value  = Mux(format =/= 0.U && strb(4 * i + 3, 4 * i).andR, chosen, 0.S).pad(34)
    Mux(signs(i), 0.S - value, value)
  }

  val beatSum = (byteLanes ++ wordLanes).reduce(_ +& _)

  // Bucket accumulator: one read port (beats or statistics) and one write port (beats or reference load).
  val buckets = SyncReadMem(p.buckets, SInt(p.accBits.W))

  val idle :: drain :: stats :: flush :: final1 :: final2 :: decide :: Nil = Enum(7)

  val state     = RegInit(idle)
  val statIndex = RegInit(0.U(p.bucketBits.W))
  val statRead  = state === stats
  val bucket    = beatAddr(p.bucketBits - 1, 0)
  val readData  = buckets.read(Mux(statRead, statIndex, bucket), statRead || (beat && inWindow))

  val s1Valid   = RegNext(beat && inWindow, false.B)
  val s1Bucket  = RegNext(bucket)
  val s1Sum     = RegNext(beatSum)
  val sameLast  = Wire(Bool())
  val last      = Reg(SInt(p.accBits.W))
  val updated   = Mux(sameLast, last, readData) +% s1Sum.pad(p.accBits)
  val loadRef   = io.config.valid && region === 1.U
  val reference = io.config.bits.data(p.accBits - 1, 0).asSInt
  val wen       = s1Valid || loadRef
  val waddr     = Mux(loadRef, offset(p.bucketBits - 1, 0), s1Bucket)
  val wdata     = Mux(loadRef, 0.S(p.accBits.W) -% reference, updated)
  when(wen)(buckets.write(waddr, wdata))
  sameLast       := RegNext(wen, false.B) && RegNext(waddr) === s1Bucket
  when(wen)(last := wdata)

  // Statistics pipeline: read -> saturate -> d^2 -> d^4.
  val maxDev  = (1 << (p.devBits - 1)) - 1
  val r0      = RegNext(statRead, false.B)
  val p1Valid = RegNext(r0, false.B)
  val p1      = RegNext(Mux(readData > maxDev.S, maxDev.S, Mux(readData < (-maxDev - 1).S, (-maxDev - 1).S, readData)))
  val p1Dev   = p1(p.devBits - 1, 0).asSInt
  val p2Valid = RegNext(p1Valid, false.B)
  val p2      = RegNext((p1Dev * p1Dev).asUInt(2 * p.devBits - 1, 0))
  val p3Valid = RegNext(p2Valid, false.B)
  val p3      = RegNext(p2 * p2)
  val high    = Reg(SInt(p.devBits.W))
  val low     = Reg(SInt(p.devBits.W))
  val s2      = Reg(UInt(p.s2Bits.W))
  val s4      = Reg(UInt(p.s4Bits.W))
  val square  = Reg(UInt((2 * p.s2Bits).W))
  val rhs     = Reg(UInt((2 * p.s2Bits + 16).W))
  val lhs     = Reg(UInt((p.s4Bits + p.bucketBits + 8).W))
  when(p1Valid) {
    high := Mux(p1Dev > high, p1Dev, high)
    low  := Mux(p1Dev < low, p1Dev, low)
  }
  when(p2Valid)(s2 := s2 + p2)
  when(p3Valid)(s4 := s4 + p3)

  val verdict = Reg(Valid(new Verdict(p)))
  verdict.valid := false.B
  io.verdict    := verdict
  io.fault      := fault
  io.fenceReady := state === idle
  when(start) {
    current := 0.U
    fault   := false.B
  }

  switch(state) {
    is(idle) {
      when(io.fence)(state := drain)
    }
    is(drain) {
      when(!awQueue.io.deq.valid && !wQueue.io.deq.valid && !s1Valid) {
        state     := Mux(scEnable, stats, decide)
        statIndex := 0.U
        high      := (-maxDev - 1).S
        low       := maxDev.S
        s2        := 0.U
        s4        := 0.U
      }
    }
    is(stats) {
      statIndex                                   := statIndex + 1.U
      when(statIndex === (p.buckets - 1).U)(state := flush)
    }
    is(flush) {
      when(!r0 && !p1Valid && !p2Valid && !p3Valid)(state := final1)
    }
    is(final1) {
      square := s2 * s2
      state  := final2
    }
    is(final2) {
      rhs   := square * kurtLimit
      lhs   := s4 << (p.bucketBits + 8)
      state := decide
    }
    is(decide) {
      val hashFail     = hash =/= expectedHash
      val rangeFail    = scEnable && (high -& low).asUInt > rangeLimit
      val kurtosisFail = scEnable && lhs > rhs
      verdict.valid             := true.B
      verdict.bits.entry        := current
      verdict.bits.hashFail     := hashFail
      verdict.bits.rangeFail    := rangeFail
      verdict.bits.kurtosisFail := kurtosisFail
      verdict.bits.hash         := hash
      verdict.bits.max          := high
      verdict.bits.min          := low
      verdict.bits.s2           := s2
      verdict.bits.s4           := s4
      fault                     := fault || hashFail || rangeFail || kurtosisFail
      current                   := current + 1.U
      hash                      := 0.U
      state                     := idle
    }
  }
}
