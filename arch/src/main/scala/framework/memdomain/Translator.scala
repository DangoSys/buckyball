package framework.memdomain

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyballConfigs.CustomBuckyballConfig
import framework.memdomain.mem.{SramReadIO, SramWriteIO}
import framework.balldomain.blink.{SramReadWithInfo, SramWriteWithInfo}

class Translator(implicit b: CustomBuckyballConfig, p: Parameters) extends Module{
  val numBanks = b.sp_banks + b.acc_banks
  class BanksIO extends Bundle {
    val sramread  = Vec(numBanks, new SramReadWithInfo(b.spad_bank_entries, b.spad_w))
    val sramwrite = Vec(numBanks, new SramWriteWithInfo(b.spad_bank_entries, b.spad_w, b.spad_mask_len))
  }

  val io = IO(new Bundle {
    val dma_in      = new BanksIO
    // val dma_return  = Flipped(new BanksIO)
    val dma_out     = Flipped(new BanksIO)
    val exec_in     = new BanksIO
    // val exec_return = Flipped(new BanksIO)
    val exec_out    = Flipped(new BanksIO)
  })

  io.dma_out     <> io.dma_in
  // io.dma_return  <> io.dma_in
  io.exec_out    <> io.exec_in
  // io.exec_return <> io.exec_in
}

