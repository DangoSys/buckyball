package framework.switcher

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyballConfigs.CustomBuckyballConfig
import framework.memdomain.mem.{SramReadIO, SramWriteIO, SramReadReq, SramReadResp, SramWriteReq}
import framework.blink.{SramReadWithRobId, SramWriteWithRobId, SramReadWithInfo, SramWriteWithInfo}

class ToVirtualLine(implicit b: CustomBuckyballConfig, p: Parameters) extends Module {
  // Total number of unified virtual banks = sp_banks + acc_banks
  private val numBanks = b.sp_banks + b.acc_banks

  val io = IO(new Bundle {
    // Physical SRAM/ACC ports
    val sramRead_i  = Vec(b.sp_banks, new SramReadWithRobId(b.spad_bank_entries, b.spad_w))
    val sramWrite_i = Vec(b.sp_banks, new SramWriteWithRobId(b.spad_bank_entries, b.spad_w, b.spad_mask_len))
    val accRead_i   = Vec(b.acc_banks, new SramReadWithRobId(b.acc_bank_entries, b.acc_w))
    val accWrite_i  = Vec(b.acc_banks, new SramWriteWithRobId(b.acc_bank_entries, b.acc_w, b.acc_mask_len))

    // Unified virtual interface
    val sramRead_o  = Vec(numBanks, Flipped(new SramReadWithInfo(b.spad_bank_entries, b.spad_w)))
    val sramWrite_o = Vec(numBanks, Flipped(new SramWriteWithInfo(b.spad_bank_entries, b.spad_w, b.spad_mask_len)))
  })

  // --------------------------------------------------------------------------
  // Default initialization for virtual output banks
  // --------------------------------------------------------------------------

  for (i <- 0 until numBanks) {
    val vr = io.sramRead_o(i)
    vr.io.req.valid  := false.B
    vr.io.req.bits   := DontCare
    vr.io.resp.ready := false.B
    vr.is_acc        := false.B
    vr.bank_id       := 0.U
    vr.rob_id        := 0.U

    val vw = io.sramWrite_o(i)
    vw.io.req.valid := false.B
    vw.io.req.bits  := DontCare
    vw.is_acc       := false.B
    vw.bank_id      := 0.U
    vw.rob_id       := 0.U
  }

  // Default init for physical inputs
  for (i <- 0 until b.sp_banks) {
    val spR = io.sramRead_i(i)
    spR.io.req.ready  := false.B
    spR.io.resp.valid := false.B
    spR.io.resp.bits  := DontCare

    val spW = io.sramWrite_i(i)
    spW.io.req.ready := false.B
  }

  for (i <- 0 until b.acc_banks) {
    val accR = io.accRead_i(i)
    accR.io.req.ready  := false.B
    accR.io.resp.valid := false.B
    accR.io.resp.bits  := DontCare

    val accW = io.accWrite_i(i)
    accW.io.req.ready := false.B
  }

  // --------------------------------------------------------------------------
  // Read Path Routing: SPAD → virtual line (low bank index range)
  // --------------------------------------------------------------------------

  for (i <- 0 until b.sp_banks) {
    val vR   = io.sramRead_o(i)
    val sp   = io.sramRead_i(i)
    val spRq = sp.io.req

    val selSp = spRq.valid

    vR.io.req.valid        := selSp
    spRq.ready             := selSp && vR.io.req.ready

    vR.io.req.bits.addr    := spRq.bits.addr
    vR.io.req.bits.fromDMA := spRq.bits.fromDMA

    vR.is_acc              := false.B
    vR.bank_id             := i.U(vR.bank_id.getWidth.W)
    vR.rob_id              := sp.rob_id

    sp.io.resp.valid       := vR.io.resp.valid
    sp.io.resp.bits        := vR.io.resp.bits
    vR.io.resp.ready       := sp.io.resp.ready && selSp
  }

  // --------------------------------------------------------------------------
  // Read Path Routing: ACC → virtual line (higher bank index range)
  // --------------------------------------------------------------------------

  for (i <- 0 until b.acc_banks) {
    val j = i + b.sp_banks
    val vR   = io.sramRead_o(j)
    val acc  = io.accRead_i(i)
    val accRq = acc.io.req

    val selAcc = accRq.valid

    vR.io.req.valid        := selAcc
    accRq.ready            := selAcc && vR.io.req.ready

    vR.io.req.bits.addr    := accRq.bits.addr
    vR.io.req.bits.fromDMA := accRq.bits.fromDMA

    vR.is_acc              := true.B
    vR.bank_id             := i.U(vR.bank_id.getWidth.W)
    vR.rob_id              := acc.rob_id

    acc.io.resp.valid      := vR.io.resp.valid
    acc.io.resp.bits       := vR.io.resp.bits
    vR.io.resp.ready       := acc.io.resp.ready && selAcc
  }

  // --------------------------------------------------------------------------
  // Write Path Routing: SPAD → virtual line
  // --------------------------------------------------------------------------

  for (i <- 0 until b.sp_banks) {
    val vW   = io.sramWrite_o(i)
    val sp   = io.sramWrite_i(i)
    val spRq = sp.io.req

    val selSp = spRq.valid

    vW.io.req.valid      := selSp
    spRq.ready           := selSp && vW.io.req.ready

    vW.io.req.bits.addr  := spRq.bits.addr
    vW.io.req.bits.data  := spRq.bits.data
    vW.io.req.bits.mask  := spRq.bits.mask

    vW.is_acc            := false.B
    vW.bank_id           := i.U(vW.bank_id.getWidth.W)
    vW.rob_id            := sp.rob_id
  }

  // --------------------------------------------------------------------------
  // Write Path Routing: ACC → virtual line
  // --------------------------------------------------------------------------

  for (i <- 0 until b.acc_banks) {
    val j = i + b.sp_banks
    val vW    = io.sramWrite_o(j)
    val acc   = io.accWrite_i(i)
    val accRq = acc.io.req

    val selAcc = accRq.valid

    vW.io.req.valid      := selAcc
    accRq.ready          := selAcc && vW.io.req.ready

    vW.io.req.bits.addr  := accRq.bits.addr
    vW.io.req.bits.data  := accRq.bits.data
    vW.io.req.bits.mask  := accRq.bits.mask

    vW.is_acc            := true.B
    vW.bank_id           := i.U(vW.bank_id.getWidth.W)
    vW.rob_id            := acc.rob_id
  }
}
