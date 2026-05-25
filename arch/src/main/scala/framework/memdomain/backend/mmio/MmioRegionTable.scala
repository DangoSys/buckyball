package framework.memdomain.backend.mmio

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.top.GlobalConfig

/**
 * MmioRegionTable: hardware binding table.
 *
 *   table[main_bank_id] = { valid, mmio_addr, size_rows }
 *
 *   alloc port  : driven by MemConfiger when mmio_set instruction fires.
 *                  size_rows == 0 means dealloc (valid := false).
 *                  size_rows  > 0 means alloc/overwrite.
 *
 *   dealloc port: driven by MemConfiger when mem_dealloc (mset with alloc=false) fires,
 *                  performing the implicit lifecycle release.
 *
 *   lookup port : vector combinational read by MmioRouter for in-flight Ball reads
 *                  (one entry per Ball, sized by ballDomain.ballNum).
 */
@instantiable
class MmioRegionTable(val b: GlobalConfig) extends Module {

  @public
  val io = IO(new Bundle {
    val alloc   = Flipped(Valid(new MmioAllocReq(b)))
    val dealloc = Flipped(Valid(UInt(log2Up(b.memDomain.bankNum).W)))

    // Vector lookup: one combinational port per Ball.
    val lookup_main_bank =
      Input(Vec(b.ballDomain.ballNum, UInt(log2Up(b.memDomain.bankNum).W)))
    val lookup_result    =
      Output(Vec(b.ballDomain.ballNum, new MmioRegionEntry(b)))
  })

  val table = RegInit(
    VecInit(
      Seq.fill(b.memDomain.bankNum)(
        0.U.asTypeOf(new MmioRegionEntry(b))
      )
    )
  )

  when(io.alloc.valid) {
    val isDealloc = io.alloc.bits.size_rows === 0.U
    when(isDealloc) {
      table(io.alloc.bits.main_bank).valid := false.B
    }.otherwise {
      table(io.alloc.bits.main_bank).valid     := true.B
      table(io.alloc.bits.main_bank).mmio_addr := io.alloc.bits.mmio_addr
      table(io.alloc.bits.main_bank).size_rows := io.alloc.bits.size_rows
    }
  }

  when(io.dealloc.valid) {
    table(io.dealloc.bits).valid := false.B
  }

  for (i <- 0 until b.ballDomain.ballNum) {
    io.lookup_result(i) := table(io.lookup_main_bank(i))
  }
}
