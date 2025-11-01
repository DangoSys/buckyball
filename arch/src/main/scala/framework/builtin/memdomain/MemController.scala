package framework.builtin.memdomain

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import framework.builtin.memdomain.mem.{SramReadIO, SramWriteIO, Scratchpad}

/**
 * MemController: Controller that encapsulates scratchpad and accumulator
 * Provides DMA interface and Ball Domain interface
 */
class MemController(implicit b: CustomBuckyBallConfig, p: Parameters) extends Module {
  val io = IO(new Bundle {
    // DMA interface - for MemLoader and MemStorer access
    val dma = new Bundle {
      val sramRead  = Vec(b.sp_banks, new SramReadIO(b.spad_bank_entries, b.spad_w))
      val sramWrite = Vec(b.sp_banks, new SramWriteIO(b.spad_bank_entries, b.spad_w, b.spad_mask_len))
      val accRead   = Vec(b.acc_banks, new SramReadIO(b.acc_bank_entries, b.acc_w))
      val accWrite  = Vec(b.acc_banks, new SramWriteIO(b.acc_bank_entries, b.acc_w, b.acc_mask_len))
    }

    // Ball Domain interface - for BallController access
    val ballDomain = new Bundle {
      val sramRead  = Vec(b.sp_banks, new SramReadIO(b.spad_bank_entries, b.spad_w))
      val sramWrite = Vec(b.sp_banks, new SramWriteIO(b.spad_bank_entries, b.spad_w, b.spad_mask_len))
      val accRead   = Vec(b.acc_banks, new SramReadIO(b.acc_bank_entries, b.acc_w))
      val accWrite  = Vec(b.acc_banks, new SramWriteIO(b.acc_bank_entries, b.acc_w, b.acc_mask_len))
    }
  })

  val spad = Module(new Scratchpad(b))

  // Connect DMA interface to Scratchpad's DMA ports
  io.dma.sramRead   <> spad.io.dma.sramread
  io.dma.sramWrite  <> spad.io.dma.sramwrite
  io.dma.accRead    <> spad.io.dma.accread
  io.dma.accWrite   <> spad.io.dma.accwrite

  // Connect Ball Domain interface to Scratchpad's execution ports
  io.ballDomain.sramRead  <> spad.io.exec.sramread
  io.ballDomain.sramWrite <> spad.io.exec.sramwrite
  io.ballDomain.accRead   <> spad.io.exec.accread
  io.ballDomain.accWrite  <> spad.io.exec.accwrite
}
