package examples.balls.mxint8gemv

import framework.generic.contract.blink.BlinkParam
import framework.generic.contract.configs.BallRouteParam
import framework.ball.sub_rob.ExpansionParam

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import framework.ball.bbus.{BallFactory, BallPorts}
import framework.ball.blink.{BlinkIO, HasBlink}

@instantiable
class Mxint8gemvBall(
  link:  BlinkParam,
  route: BallRouteParam)
    extends Module
    with HasBlink {
  val ball                                                                                                                                                  = route
  require(link.bankWidth == 128 && link.bankEntries == 1024)
  require(ball.inBW == 5 && ball.outBW == 1)
  @public val blink                                                                                                                                         = IO(new BlinkIO(link, route))
  val idle :: oldReq :: oldResp :: load :: mac :: selectScale :: align :: add :: normalize :: advance :: encode :: storeReq :: storeResp :: complete :: Nil =
    Enum(14)
  val state                                                                                                                                                 = RegInit(idle)
  val rob                                                                                                                                                   = Reg(UInt(link.robIdBits.W))
  val banks                                                                                                                                                 = Reg(Vec(4, UInt(link.bankIdBits.W))) // A, B, new C, old C.
  val groups                                                                                                                                                = Reg(Vec(3, UInt(link.groupIdBits.W)))
  val bases                                                                                                                                                 = Reg(Vec(4, UInt(14.W)))
  val blocks                                                                                                                                                = Reg(UInt(11.W))
  val stride                                                                                                                                                = Reg(UInt(14.W))
  val block                                                                                                                                                 = Reg(UInt(11.W))
  val count                                                                                                                                                 = Reg(UInt(11.W))
  val scale                                                                                                                                                 = Reg(SInt(11.W))
  val valid                                                                                                                                                 = Reg(Bool())
  val emitQuantized                                                                                                                                         = Reg(Bool())
  val accumulates                                                                                                                                           = Reg(Bool())
  val values                                                                                                                                                = Reg(Vec(32, SInt(64.W)))
  val products                                                                                                                                              = Reg(Vec(32, SInt(21.W)))
  val aWords                                                                                                                                                = Reg(Vec(2, Vec(2, UInt(128.W))))
  val aExp                                                                                                                                                  = Reg(UInt(128.W))
  val bWords                                                                                                                                                = Reg(Vec(2, Vec(32, Vec(2, UInt(128.W)))))
  val bExp                                                                                                                                                  = Reg(Vec(32, UInt(128.W)))
  val issued                                                                                                                                                = Reg(Vec(5, UInt(6.W)))
  val received                                                                                                                                              = Reg(Vec(5, UInt(6.W)))
  val step                                                                                                                                                  = Reg(UInt(3.W))
  val row                                                                                                                                                   = Reg(UInt(5.W))
  val nextScale                                                                                                                                             = Reg(SInt(11.W))
  val codes                                                                                                                                                 = Reg(Vec(32, UInt(8.W)))
  val outExponent                                                                                                                                           = Reg(UInt(8.W))
  val lane                                                                                                                                                  = Reg(UInt(5.W))

  def rne(value: SInt, distance: UInt): SInt = {
    val magnitude = Mux(value < 0.S, -value, value).asUInt
    val shift     = distance(5, 0)
    val quotient  = magnitude >> shift
    val mask      = ((1.U(65.W) << shift) - 1.U)(63, 0)
    val remainder = magnitude & mask
    val half      = (1.U(64.W) << (shift - 1.U)(5, 0))(63, 0)
    val rounded   = (quotient + (remainder > half || (remainder === half && quotient(0))).asUInt)(63, 0).asSInt
    Mux(distance === 0.U, value, Mux(distance >= 64.U, 0.S(64.W), Mux(value < 0.S, -rounded, rounded)))
  }

  val currentBuffer  = Reg(Bool())
  val loadBuffer     = Reg(Bool())
  val loadBlock      = Reg(UInt(11.W))
  val prefetchActive = RegInit(false.B)
  val loading        = prefetchActive && state =/= oldReq && state =/= oldResp
  val expRefresh     = loadBlock(3, 0) === 0.U
  val requests       = Mux(expRefresh, 24.U, 16.U)
  val aRequests      = Mux(expRefresh, 3.U, 2.U)
  blink.cmdReq.ready        := state === idle
  blink.cmdResp.valid       := state === complete
  blink.cmdResp.bits.rob_id := rob
  blink.status.idle         := state === idle
  blink.status.running      := state =/= idle && state =/= complete
  for (p <- 0 until 5) {
    val read = blink.bankRead(p)
    read.rob_id       := rob
    read.bank_id      := (if (p == 0) Mux(state === oldReq || state === oldResp, banks(3), banks(0)) else banks(1))
    read.group_id     := (if (p == 0) groups(0) else groups(1) + (p - 1).U)
    if (p == 0) { when(state === oldReq || state === oldResp)(read.group_id := groups(2)) }
    read.io.req.valid := loading && issued(p) < (if (p == 0) aRequests else requests)
    val index  = issued(p)
    val column = Mux(index < 16.U, index >> 1, index - 16.U)
    read.io.req.bits.addr := (if (p == 0)
                                Mux(index < 2.U, bases(0) + (loadBlock << 1) + index, bases(1) + (loadBlock >> 4))
                              else bases(2) + column * stride + Mux(
                                index < 16.U,
                                (loadBlock << 1) + index(0),
                                (blocks << 1) + (loadBlock >> 4)
                              ))
    read.io.resp.ready    := loading
    when(loading) {
      when(read.io.req.fire)(issued(p) := issued(p) + 1.U)
      when(read.io.resp.fire) {
        val index = received(p)
        val data  = read.io.resp.bits.data
        received(p) := index + 1.U
        if (p == 0) {
          when(index < 2.U)(aWords(loadBuffer)(index(0)) := data).otherwise(aExp := data)
        } else {
          when(index < 16.U)(bWords(loadBuffer)(((p - 1) * 8).U + (index >> 1))(index(0)) := data)
            .otherwise(bExp(((p - 1) * 8).U + index - 16.U) := data)
        }
      }
    }
  }
  val write = blink.bankWrite(0)
  write.rob_id           := rob
  write.bank_id          := banks(2)
  write.group_id         := groups(2)
  write.io.req.valid     := state === storeReq
  write.io.req.bits.addr := bases(3) + row
  val header    = Cat(0.U(88.W), Cat(0.U(7.W), valid), scale.pad(16).asUInt, count.pad(16))
  val valuePair = Cat(values((row - 1.U) * 2.U + 1.U).asUInt, values((row - 1.U) * 2.U).asUInt)
  write.io.req.bits.data := Mux(
    emitQuantized,
    Mux(row === 0.U, Cat(codes.take(16).reverse), Mux(row === 1.U, Cat(codes.drop(16).reverse), outExponent.pad(128))),
    Mux(row === 0.U, header, valuePair)
  )
  write.io.req.bits.mask := VecInit(Seq.fill(link.bankMaskLen)(true.B))
  write.io.resp.ready    := state === storeResp
  val aExponent    = (aExp >> (block(3, 0) << 3))(7, 0)
  val scales       = Reg(Vec(32, SInt(11.W)))
  val loadedScales = Wire(Vec(32, SInt(11.W)))
  for (j <- 0 until 32) {
    val e = (bExp(j) >> (block(3, 0) << 3))(7, 0)
    loadedScales(j) := aExponent.zext.pad(11) + e.zext.pad(11) - 254.S
  }
  val incoming =
    (0 until 32).map(j => Mux(products(j) =/= 0.S, scales(j), -512.S(11.W))).reduce((a, b) => Mux(a > b, a, b))
  val magnitude = values.map(v => Mux(v < 0.S, -v, v).asUInt).reduce((a, b) => Mux(a > b, a, b))
  val highest   = 63.U - PriorityEncoder(Reverse(magnitude))
  val q         = highest.zext + scale - 38.S
  val boundedQ  = Mux(q < -127.S, -127.S, Mux(q > 127.S, 127.S, q))

  switch(state) {
    is(idle) {
      when(blink.cmdReq.fire) {
        val cmd          = blink.cmdReq.bits
        val k            = cmd.iter
        val packedStride = (k >> 4) + ((k +& 511.U) >> 9)
        assert(k > 0.U && k(4, 0) === 0.U && cmd.rs2(63, 58) === 0.U)
        assert(cmd.op1_col === 1.U && cmd.op2_col === 4.U && cmd.wr_col === 1.U)
        assert(cmd.op1_bank =/= cmd.op2_bank && cmd.op1_bank =/= cmd.wr_bank && cmd.op2_bank =/= cmd.wr_bank)
        assert(cmd.rs2(13, 0) +& (k >> 4) <= 1024.U)
        assert(cmd.rs2(27, 14) +& ((k +& 511.U) >> 9) <= 1024.U)
        assert(cmd.rs2(41, 28) +& (packedStride << 3) <= 1024.U)
        assert(cmd.rs2(55, 42) +& 17.U <= 1024.U)
        rob            := cmd.rob_id
        banks          := VecInit(cmd.op1_bank, cmd.op2_bank, cmd.wr_bank, cmd.wr_read_bank)
        groups         := VecInit(cmd.op1_group, cmd.op2_group, cmd.wr_group)
        bases          := VecInit(cmd.rs2(13, 0), cmd.rs2(27, 14), cmd.rs2(41, 28), cmd.rs2(55, 42))
        blocks         := k >> 5
        stride         := packedStride
        block          := 0.U
        currentBuffer  := false.B
        loadBuffer     := false.B
        loadBlock      := 0.U
        prefetchActive := true.B
        count          := 0.U
        scale          := 0.S
        valid          := false.B
        emitQuantized  := cmd.rs2(57)
        accumulates    := cmd.rs2(56)
        values         := VecInit(Seq.fill(32)(0.S(64.W)))
        issued         := VecInit(Seq.fill(5)(0.U(6.W)))
        received       := VecInit(Seq.fill(5)(0.U(6.W)))
        row            := 0.U
        state          := Mux(cmd.rs2(56), oldReq, load)
      }
    }
    is(oldReq) {
      blink.bankRead(0).io.req.valid            := true.B
      blink.bankRead(0).io.req.bits.addr        := bases(3) + row
      when(blink.bankRead(0).io.req.fire)(state := oldResp)
    }
    is(oldResp) {
      blink.bankRead(0).io.resp.ready := true.B
      when(blink.bankRead(0).io.resp.fire) {
        val data = blink.bankRead(0).io.resp.bits.data
        when(row === 0.U) {
          assert(data(127, 33) === 0.U && data(15, 0) > 0.U && data(15, 0) +& blocks <= 1024.U)
          count := data(15, 0)
          scale := data(31, 16).asSInt
          valid := data(32)
        }.otherwise {
          values((row - 1.U) * 2.U)       := data(63, 0).asSInt
          values((row - 1.U) * 2.U + 1.U) := data(127, 64).asSInt
        }
        row := row + 1.U
        state := Mux(row === 16.U, load, oldReq)
      }
    }
    is(load) {
      when(received(0) === aRequests && received.drop(1).map(_ === requests).reduce(_ && _)) {
        when(block === 0.U && accumulates) {
          assert(Mux(
            valid,
            scale >= -254.S && scale <= 254.S && values.map(_ =/= 0.S).reduce(_ || _),
            scale === 0.S && values.map(_ === 0.S).reduce(_ && _)
          ))
          for (j <- 0 until 32) {
            assert(Mux(values(j) < 0.S, -values(j), values(j)).asUInt <= (count << 51))
          }
        }
        assert(aExponent =/= 255.U)
        for (j <- 0 until 32) { assert((bExp(j) >> (block(3, 0) << 3))(7, 0) =/= 255.U) }
        scales                     := loadedScales
        when(block + 1.U < blocks) {
          loadBlock  := block + 1.U
          loadBuffer := !currentBuffer
          issued     := VecInit(Seq.fill(5)(0.U(6.W)))
          received   := VecInit(Seq.fill(5)(0.U(6.W)))
        }.otherwise(prefetchActive := false.B)
        products                   := VecInit(Seq.fill(32)(0.S(21.W)))
        step                       := 0.U
        state                      := mac
      }
    }
    is(mac) {
      for (j <- 0 until 32) {
        val a   = aWords(currentBuffer)(step(2)) >> (step(1, 0) << 5)
        val b   = bWords(currentBuffer)(j)(step(2)) >> (step(1, 0) << 5)
        val dot = (0 until 4).map(i => a(i * 8 + 7, i * 8).asSInt * b(i * 8 + 7, i * 8).asSInt).reduce(_ +& _)
        products(j) := products(j) + dot
      }
      step := step + 1.U
      when(step === 7.U)(state := selectScale)
    }
    is(selectScale) {
      nextScale := Mux(valid && scale > incoming, scale, incoming)
      state     := Mux(incoming === -512.S, advance, align)
    }
    is(align) {
      when(valid && nextScale > scale) {
        for (j <- 0 until 32) { values(j) := rne(values(j), (nextScale - scale).asUInt) }
      }
      scale := nextScale
      state := add
    }
    is(add) {
      for (j <- 0 until 32) {
        val distance = 32.S(12.W) + scales(j) - nextScale
        val term     = Mux(
          distance >= 0.S,
          (products(j).pad(64) << distance.asUInt(5, 0))(63, 0).asSInt,
          rne(products(j).pad(64), (-distance).asUInt)
        )
        when(products(j) =/= 0.S)(values(j) := values(j) + term)
      }
      state := normalize
    }
    is(normalize) {
      val nonzero = values.map(_ =/= 0.S).reduce(_ || _)
      valid                := nonzero
      when(!nonzero)(scale := 0.S)
      state                := advance
    }
    is(advance) {
      when(block + 1.U === blocks) {
        assert(count +& blocks <= 1024.U)
        count       := count + blocks
        row         := 0.U
        lane        := 0.U
        outExponent := Mux(magnitude === 0.U, 127.U, (boundedQ + 127.S).asUInt)
        state       := Mux(emitQuantized, encode, storeReq)
      }.otherwise {
        block         := block + 1.U
        currentBuffer := !currentBuffer
        state         := load
      }
    }
    is(encode) {
      val distance = outExponent.zext - 127.S - scale + 32.S
      val v        = values(lane)
      val abs      = Mux(v < 0.S, -v, v).asUInt
      val left     = (-distance).asUInt
      val overflow = left > 6.U || abs > (127.U >> left(5, 0))
      val shifted  = Mux(
        distance >= 0.S,
        rne(v, distance.asUInt),
        Mux(overflow, Mux(v < 0.S, -127.S(64.W), Mux(v > 0.S, 127.S(64.W), 0.S(64.W))), (v << left(5, 0))(63, 0).asSInt)
      )
      codes(lane)               := Mux(shifted < -127.S, -127.S, Mux(shifted > 127.S, 127.S, shifted)).asUInt
      lane                      := lane + 1.U
      when(lane === 31.U)(state := storeReq)
    }
    is(storeReq)(when(write.io.req.fire)(state := storeResp))
    is(storeResp) {
      when(write.io.resp.fire) {
        assert(write.io.resp.bits.ok)
        row   := row + 1.U
        state := Mux(row === Mux(emitQuantized, 2.U, 16.U), complete, storeReq)
      }
    }
    is(complete)(when(blink.cmdResp.fire)(state := idle))
  }
}

object Mxint8gemvBall extends BallFactory {

  def instantiate(
    link:      BlinkParam,
    route:     BallRouteParam,
    expansion: Option[ExpansionParam]
  ): BallPorts =
    BallPorts(Instantiate(new Mxint8gemvBall(link, route)).blink, None)

}
