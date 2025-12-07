package framework.switcher

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyballConfigs.CustomBuckyballConfig
import framework.memdomain.mem.{SramReadIO, SramWriteIO}
import framework.balldomain.blink.{SramReadWithInfo, SramWriteWithInfo}

class ToPhysicalLine(implicit b: CustomBuckyballConfig, p: Parameters) extends Module {

  private val numBanks = b.sp_banks + b.acc_banks

  val io = IO(new Bundle {
    // Unified virtual input ports (from ToVirtualLine)
    val sramRead_i  = Vec(numBanks, new SramReadWithInfo(b.spad_bank_entries, b.spad_w))
    val sramWrite_i = Vec(numBanks, new SramWriteWithInfo(b.spad_bank_entries, b.spad_w, b.spad_mask_len))

    // Physical memory endpoints
    val sramRead_o  = Vec(b.sp_banks, Flipped(new SramReadIO(b.spad_bank_entries, b.spad_w)))
    val sramWrite_o = Vec(b.sp_banks, Flipped(new SramWriteIO(b.spad_bank_entries, b.spad_w, b.spad_mask_len)))

    val accRead_o   = Vec(b.acc_banks, Flipped(new SramReadIO(b.acc_bank_entries, b.acc_w)))
    val accWrite_o  = Vec(b.acc_banks, Flipped(new SramWriteIO(b.acc_bank_entries, b.acc_w, b.acc_mask_len)))
  })

  // --------------------------------------------------------------------------
  // Default initialization for all physical ports
  // --------------------------------------------------------------------------

  // SPAD read/write ports
  for (i <- 0 until b.sp_banks) {
    val spR = io.sramRead_o(i)
    spR.req.valid  := false.B
    spR.req.bits   := DontCare
    spR.resp.ready := false.B

    val spW = io.sramWrite_o(i)
    spW.req.valid := false.B
    spW.req.bits  := DontCare
  }

  // ACC read/write ports
  for (i <- 0 until b.acc_banks) {
    val accR = io.accRead_o(i)
    accR.req.valid  := false.B
    accR.req.bits   := DontCare
    accR.resp.ready := false.B

    val accW = io.accWrite_o(i)
    accW.req.valid := false.B
    accW.req.bits  := DontCare
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
    spR.req.valid        := vR.io.req.valid
    spR.req.bits.addr    := vR.io.req.bits.addr
    spR.req.bits.fromDMA := vR.io.req.bits.fromDMA

    vR.io.req.ready         := spR.req.ready

    // Response path (SPAD → virtual)
    vR.io.resp.valid        := spR.resp.valid
    vR.io.resp.bits         := spR.resp.bits
    spR.resp.ready       := vR.io.resp.ready
  }

  // --------------------------------------------------------------------------
  // Read routing: virtual → ACC  (indices sp_banks .. sp_banks+acc_banks-1)
  // --------------------------------------------------------------------------

  for (i <- 0 until b.acc_banks) {
    val idx  = i + b.sp_banks
    val vR   = io.sramRead_i(idx)
    val accR = io.accRead_o(i)

    // Request path (virtual → ACC)
    accR.req.valid        := vR.io.req.valid
    accR.req.bits.addr    := vR.io.req.bits.addr
    accR.req.bits.fromDMA := vR.io.req.bits.fromDMA

    vR.io.req.ready          := accR.req.ready

    // Response path (ACC → virtual)
    vR.io.resp.valid         := accR.resp.valid
    vR.io.resp.bits          := accR.resp.bits
    accR.resp.ready       := vR.io.resp.ready
  }

  // --------------------------------------------------------------------------
  // Write routing: virtual → SPAD
  // --------------------------------------------------------------------------

  for (i <- 0 until b.sp_banks) {
    val vW  = io.sramWrite_i(i)
    val spW = io.sramWrite_o(i)

    spW.req.valid       := vW.io.req.valid
    spW.req.bits.addr   := vW.io.req.bits.addr
    spW.req.bits.data   := vW.io.req.bits.data
    spW.req.bits.mask   := vW.io.req.bits.mask

    vW.io.req.ready        := spW.req.ready
  }

  // --------------------------------------------------------------------------
  // Write routing: virtual → ACC
  // --------------------------------------------------------------------------

  for (i <- 0 until b.acc_banks) {
    val idx  = i + b.sp_banks
    val vW   = io.sramWrite_i(idx)
    val accW = io.accWrite_o(i)

    accW.req.valid       := vW.io.req.valid
    accW.req.bits.addr   := vW.io.req.bits.addr
    accW.req.bits.data   := vW.io.req.bits.data
    accW.req.bits.mask   := vW.io.req.bits.mask

    vW.io.req.ready         := accW.req.ready
  }
}