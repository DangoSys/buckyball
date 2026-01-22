package framework.memdomain.backend.accpipe

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}

import framework.top.GlobalConfig
import framework.memdomain.backend.banks.{SramReadIO, SramWriteIO}

/**
 * AccPipe: Accumulator Pipeline
 * - Direct write (wmode=0): write.req -> bank write -> forward resp
 * - Accum write (wmode=1): bank read -> (old + new with mask) -> bank write -> forward resp
 * - Read: bank read -> forward resp
 *
 * This version:
 * - Uses correct IO directions based on your SramReadIO/SramWriteIO definitions
 * - Uses strict Decoupled handshakes
 * - Latches op type/address/data/mask
 * - Latches old_data on read resp fire (no cross-state resp.bits usage)
 */
@instantiable
class AccPipe(val b: GlobalConfig) extends Module {

  @public
  val io = IO(new Bundle {
    // Interface to SramBank
    // Your SramReadIO/SramWriteIO are SLAVE-shaped (req is Flipped), so master must Flipped(...)
    val sramRead  = Flipped(new SramReadIO(b))  // AccPipe -> bank: req out, resp in
    val sramWrite = Flipped(new SramWriteIO(b)) // AccPipe -> bank: req out, resp in

    // Interface from midend (AccPipe is slave)
    val read  = new SramReadIO(b)  // midend -> AccPipe: req in, resp out
    val write = new SramWriteIO(b) // midend -> AccPipe: req in, resp out

    // Control and status signals
    val bank_id        = Input(UInt(log2Up(b.memDomain.bankNum).W))
    // in AccPipe IO bundle, add:
    val target_bank_id = Output(UInt(log2Up(b.memDomain.bankNum).W))

    val busy = Output(Bool())
  })

  // ---------------------------------------------------------------------------
  // Latched transaction context
  // ---------------------------------------------------------------------------
  val curBankId = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  // target bank id for routing (combinational)

  val opIsRead  = RegInit(false.B) // true: read transaction; false: write transaction
  val opIsAccum = RegInit(false.B) // true only for write + wmode=1

  val latAddr = RegInit(0.U(log2Ceil(b.memDomain.bankEntries).W))
  val latData = RegInit(0.U(b.memDomain.bankWidth.W))
  val latMask = RegInit(VecInit(Seq.fill(b.memDomain.bankMaskLen)(false.B)))

  val oldData = RegInit(0.U(b.memDomain.bankWidth.W))
  val resData = RegInit(0.U(b.memDomain.bankWidth.W))

  // ---------------------------------------------------------------------------
  // FSM
  // ---------------------------------------------------------------------------
  val s_idle :: s_wait_read_resp :: s_issue_write_back :: s_wait_write_resp :: s_wait_direct_write_resp :: Nil =
    Enum(5)
  val state                                                                                                    = RegInit(s_idle)

  io.target_bank_id := Mux(state === s_idle, io.bank_id, curBankId)

  io.busy := (state =/= s_idle)

  // ---------------------------------------------------------------------------
  // Defaults (avoid multi-drive)
  // ---------------------------------------------------------------------------
  // Midend side defaults
  io.read.req.ready      := false.B
  io.read.resp.valid     := false.B
  io.read.resp.bits.data := 0.U

  io.write.req.ready    := false.B
  io.write.resp.valid   := false.B
  io.write.resp.bits.ok := false.B

  // Bank side defaults
  io.sramRead.req.valid     := false.B
  io.sramRead.req.bits.addr := 0.U
  io.sramRead.resp.ready    := false.B

  io.sramWrite.req.valid      := false.B
  io.sramWrite.req.bits.addr  := 0.U
  io.sramWrite.req.bits.data  := 0.U
  io.sramWrite.req.bits.mask  := VecInit(Seq.fill(b.memDomain.bankMaskLen)(false.B))
  io.sramWrite.req.bits.wmode := false.B // IMPORTANT: write-back is a normal write
  io.sramWrite.resp.ready     := false.B

  // ---------------------------------------------------------------------------
  // Helpers: masked accumulate (segment-wise)
  // ---------------------------------------------------------------------------
  private def maskedAccumulate(oldU: UInt, addU: UInt, maskV: Vec[Bool]): UInt = {
    val segW = b.memDomain.bankWidth / b.memDomain.bankMaskLen
    // Optional safety: if not divisible, elaboration will still succeed but behavior is wrong.
    // You can assert if you want:
    // require(b.memDomain.bankWidth % b.memDomain.bankMaskLen == 0, "bankWidth must be divisible by bankMaskLen")

    val oldVec = oldU.asTypeOf(Vec(b.memDomain.bankMaskLen, UInt(segW.W)))
    val addVec = addU.asTypeOf(Vec(b.memDomain.bankMaskLen, UInt(segW.W)))

    val resVec = Wire(Vec(b.memDomain.bankMaskLen, UInt(segW.W)))
    for (i <- 0 until b.memDomain.bankMaskLen) {
      resVec(i) := Mux(maskV(i), oldVec(i) + addVec(i), oldVec(i))
    }
    resVec.asUInt
  }

  // ---------------------------------------------------------------------------
  // FSM behavior
  // ---------------------------------------------------------------------------
  switch(state) {
    is(s_idle) {
      // Priority: write first, then read.
      // IMPORTANT: never make req.valid depend on ready/fire; that can create combinational loops.
      val writeValid  = io.write.req.valid
      val writeAccum  = writeValid && io.write.req.bits.wmode
      val writeDirect = writeValid && !io.write.req.bits.wmode
      val readValid   = !writeValid && io.read.req.valid

      // Downstream bank requests (valid depends only on upstream valid + state)
      io.sramRead.req.valid     := writeAccum || readValid
      io.sramRead.req.bits.addr := Mux(writeAccum, io.write.req.bits.addr, io.read.req.bits.addr)

      io.sramWrite.req.valid      := writeDirect
      io.sramWrite.req.bits.addr  := io.write.req.bits.addr
      io.sramWrite.req.bits.data  := io.write.req.bits.data
      io.sramWrite.req.bits.mask  := io.write.req.bits.mask
      io.sramWrite.req.bits.wmode := false.B

      // Upstream ready is a function of the selected downstream ready
      io.write.req.ready := Mux(
        writeAccum,
        io.sramRead.req.ready,
        Mux(writeDirect, io.sramWrite.req.ready, false.B)
      )
      io.read.req.ready  := readValid && io.sramRead.req.ready

      // State transitions + latching on fire
      when(writeAccum && io.sramRead.req.fire) {
        curBankId := io.bank_id
        opIsRead  := false.B
        opIsAccum := true.B
        latAddr   := io.write.req.bits.addr
        latData   := io.write.req.bits.data
        latMask   := io.write.req.bits.mask
        state     := s_wait_read_resp
      }.elsewhen(writeDirect && io.sramWrite.req.fire) {
        curBankId := io.bank_id
        opIsRead  := false.B
        opIsAccum := false.B
        latAddr   := io.write.req.bits.addr
        latData   := io.write.req.bits.data
        latMask   := io.write.req.bits.mask
        state     := s_wait_direct_write_resp
      }.elsewhen(readValid && io.sramRead.req.fire) {
        curBankId := io.bank_id
        opIsRead  := true.B
        opIsAccum := false.B
        latAddr   := io.read.req.bits.addr
        state     := s_wait_read_resp
      }
    }

    is(s_wait_read_resp) {
      when(opIsRead) {
        // Pure read: directly forward bank resp -> midend resp with proper handshake
        io.read.resp.valid     := io.sramRead.resp.valid
        io.read.resp.bits.data := io.sramRead.resp.bits.data
        io.sramRead.resp.ready := io.read.resp.ready

        when(io.sramRead.resp.fire) {
          state := s_idle
        }
      }.elsewhen(opIsAccum) {
        // Accum path: always accept the bank resp (store old data), no dependency on midend ready here
        io.sramRead.resp.ready := true.B
        when(io.sramRead.resp.fire) {
          oldData := io.sramRead.resp.bits.data
          // Compute result right away and latch
          resData := maskedAccumulate(io.sramRead.resp.bits.data, latData, latMask)
          state   := s_issue_write_back
        }
      }.otherwise {
        // Should not happen
        state := s_idle
      }
    }

    is(s_issue_write_back) {
      // Issue bank write-back for accumulated result
      io.sramWrite.req.valid      := true.B
      io.sramWrite.req.bits.addr  := latAddr
      io.sramWrite.req.bits.data  := resData
      io.sramWrite.req.bits.mask  := latMask
      io.sramWrite.req.bits.wmode := false.B // IMPORTANT: already accumulated in AccPipe

      when(io.sramWrite.req.fire) {
        state := s_wait_write_resp
      }
    }

    is(s_wait_write_resp) {
      // Forward bank write resp to midend write resp
      io.write.resp.valid     := io.sramWrite.resp.valid
      io.write.resp.bits.ok   := io.sramWrite.resp.bits.ok
      io.sramWrite.resp.ready := io.write.resp.ready

      when(io.sramWrite.resp.fire) {
        // Done
        opIsAccum := false.B
        state     := s_idle
      }
    }

    is(s_wait_direct_write_resp) {
      // Direct write: forward resp
      io.write.resp.valid     := io.sramWrite.resp.valid
      io.write.resp.bits.ok   := io.sramWrite.resp.bits.ok
      io.sramWrite.resp.ready := io.write.resp.ready

      when(io.sramWrite.resp.fire) {
        state := s_idle
      }
    }
  }
}
