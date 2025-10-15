package framework.builtin.memdomain

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import framework.builtin.memdomain.mem.{SramReadIO, SramWriteIO, Scratchpad}

/**
 * MemController: 封装了scratchpad和accumulator的控制器
 * 提供DMA接口和Ball Domain接口
 */
class MemController(implicit b: CustomBuckyBallConfig, p: Parameters) extends Module {
  val io = IO(new Bundle {
    // DMA接口 - 用于MemLoader和MemStorer访问
    val dma = new Bundle {
      val sramRead  = Vec(b.sp_banks, new SramReadIO(b.spad_bank_entries, b.spad_w))
      val sramWrite = Vec(b.sp_banks, new SramWriteIO(b.spad_bank_entries, b.spad_w, b.spad_mask_len))
      val accRead   = Vec(b.acc_banks, new SramReadIO(b.acc_bank_entries, b.acc_w))
      val accWrite  = Vec(b.acc_banks, new SramWriteIO(b.acc_bank_entries, b.acc_w, b.acc_mask_len))
    }

    // Ball Domain接口 - 用于BallController访问
    val ballDomain = new Bundle {
      val sramRead  = Vec(b.sp_banks, new SramReadIO(b.spad_bank_entries, b.spad_w))
      val sramWrite = Vec(b.sp_banks, new SramWriteIO(b.spad_bank_entries, b.spad_w, b.spad_mask_len))
      val accRead   = Vec(b.acc_banks, new SramReadIO(b.acc_bank_entries, b.acc_w))
      val accWrite  = Vec(b.acc_banks, new SramWriteIO(b.acc_bank_entries, b.acc_w, b.acc_mask_len))
    }
  })

  val spad = Module(new Scratchpad(b))

  // 连接DMA接口到Scratchpad的DMA端口
  io.dma.sramRead   <> spad.io.dma.sramread
  io.dma.sramWrite  <> spad.io.dma.sramwrite
  io.dma.accRead    <> spad.io.dma.accread
  io.dma.accWrite   <> spad.io.dma.accwrite

  // 连接Ball Domain接口到Scratchpad的执行端口
  io.ballDomain.sramRead  <> spad.io.exec.sramread
  io.ballDomain.sramWrite <> spad.io.exec.sramwrite
  io.ballDomain.accRead   <> spad.io.exec.accread
  io.ballDomain.accWrite  <> spad.io.exec.accwrite
}
