package framework.switcher

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyballConfigs.CustomBuckyballConfig
import framework.memdomain.mem.{SramReadIO, SramWriteIO}
import framework.blink.{SramReadWithRobId, SramWriteWithRobId, SramReadWithInfo, SramWriteWithInfo}

class ToPhysicalLine(implicit b: CustomBuckyballConfig, p: Parameters) extends Module {

  private val numBanks = b.sp_banks + b.acc_banks

  val io = IO(new Bundle {
    // Unified virtual input ports (from ToVirtualLine)
    val sramRead_i  = Vec(numBanks, new SramReadWithInfo(b.spad_bank_entries, b.spad_w))
    val sramWrite_i = Vec(numBanks, new SramWriteWithInfo(b.spad_bank_entries, b.spad_w, b.spad_mask_len))

    // Physical memory endpoints
    val sramRead_o  = Vec(b.sp_banks, Flipped(new SramReadWithRobId(b.spad_bank_entries, b.spad_w)))
    val sramWrite_o = Vec(b.sp_banks, Flipped(new SramWriteWithRobId(b.spad_bank_entries, b.spad_w, b.spad_mask_len)))

    val accRead_o   = Vec(b.acc_banks, Flipped(new SramReadWithRobId(b.acc_bank_entries, b.acc_w)))
    val accWrite_o  = Vec(b.acc_banks, Flipped(new SramWriteWithRobId(b.acc_bank_entries, b.acc_w, b.acc_mask_len)))
  })

  // --------------------------------------------------------------------------
  // Default initialization for all physical ports
  // --------------------------------------------------------------------------

  // SPAD read/write ports
  for (i <- 0 until b.sp_banks) {
    val spR = io.sramRead_o(i)
    spR.io.req.valid  := false.B
    spR.io.req.bits   := DontCare
    spR.io.resp.ready := false.B
    spR.rob_id        := 0.U

    val spW = io.sramWrite_o(i)
    spW.io.req.valid := false.B
    spW.io.req.bits  := DontCare
    spW.rob_id       := 0.U
  }

  // ACC read/write ports
  for (i <- 0 until b.acc_banks) {
    val accR = io.accRead_o(i)
    accR.io.req.valid  := false.B
    accR.io.req.bits   := DontCare
    accR.io.resp.ready := false.B
    accR.rob_id        := 0.U

    val accW = io.accWrite_o(i)
    accW.io.req.valid := false.B
    accW.io.req.bits  := DontCare
    accW.rob_id       := 0.U
  }

  // Default values for all virtual ports
  for (i <- 0 until numBanks) {
    val vR = io.sramRead_i(i)
    vR.io.req.ready  := false.B
    vR.io.resp.valid := false.B
    vR.io.resp.bits  := DontCare

    val vW = io.sramWrite_i(i)
    vW.io.req.ready := false.B
  }

  // --------------------------------------------------------------------------
  // Read routing: virtual → SPAD   (indices 0 .. sp_banks-1)
  // --------------------------------------------------------------------------

  for (i <- 0 until b.sp_banks) {
    val vR  = io.sramRead_i(i)
    val spR = io.sramRead_o(i)

    // Request path (virtual → SPAD)
    spR.io.req.valid        := vR.io.req.valid
    spR.io.req.bits.addr    := vR.io.req.bits.addr
    spR.io.req.bits.fromDMA := vR.io.req.bits.fromDMA
    spR.rob_id              := vR.rob_id

    vR.io.req.ready         := spR.io.req.ready

    // Response path (SPAD → virtual)
    vR.io.resp.valid        := spR.io.resp.valid
    vR.io.resp.bits         := spR.io.resp.bits
    spR.io.resp.ready       := vR.io.resp.ready
  }

  // --------------------------------------------------------------------------
  // Read routing: virtual → ACC  (indices sp_banks .. sp_banks+acc_banks-1)
  // --------------------------------------------------------------------------

  for (i <- 0 until b.acc_banks) {
    val idx  = i + b.sp_banks
    val vR   = io.sramRead_i(idx)
    val accR = io.accRead_o(i)

    // Request path (virtual → ACC)
    accR.io.req.valid        := vR.io.req.valid
    accR.io.req.bits.addr    := vR.io.req.bits.addr
    accR.io.req.bits.fromDMA := vR.io.req.bits.fromDMA
    accR.rob_id              := vR.rob_id

    vR.io.req.ready          := accR.io.req.ready

    // Response path (ACC → virtual)
    vR.io.resp.valid         := accR.io.resp.valid
    vR.io.resp.bits          := accR.io.resp.bits
    accR.io.resp.ready       := vR.io.resp.ready
  }

  // --------------------------------------------------------------------------
  // Write routing: virtual → SPAD
  // --------------------------------------------------------------------------

  for (i <- 0 until b.sp_banks) {
    val vW  = io.sramWrite_i(i)
    val spW = io.sramWrite_o(i)

    spW.io.req.valid       := vW.io.req.valid
    spW.io.req.bits.addr   := vW.io.req.bits.addr
    spW.io.req.bits.data   := vW.io.req.bits.data
    spW.io.req.bits.mask   := vW.io.req.bits.mask
    spW.rob_id             := vW.rob_id

    vW.io.req.ready        := spW.io.req.ready
  }

  // --------------------------------------------------------------------------
  // Write routing: virtual → ACC
  // --------------------------------------------------------------------------

  for (i <- 0 until b.acc_banks) {
    val idx  = i + b.sp_banks
    val vW   = io.sramWrite_i(idx)
    val accW = io.accWrite_o(i)

    accW.io.req.valid       := vW.io.req.valid
    accW.io.req.bits.addr   := vW.io.req.bits.addr
    accW.io.req.bits.data   := vW.io.req.bits.data
    accW.io.req.bits.mask   := vW.io.req.bits.mask
    accW.rob_id             := vW.rob_id

    vW.io.req.ready         := accW.io.req.ready
  }
}
