package framework.memdomain.backend.banks.btrace

import chisel3._
import chisel3.util.HasBlackBoxResource

class AccessWriteDPI extends BlackBox with HasBlackBoxResource {

  val io = IO(new Bundle {
    val clock       = Input(Clock())
    val reset       = Input(Bool())
    val fire        = Input(Bool())
    val idle        = Input(Bool())
    val stream_hart = Input(UInt(64.W))
    val hart        = Input(UInt(64.W))
    val inst        = Input(UInt(64.W))
    val shared      = Input(UInt(32.W))
    val physical    = Input(UInt(32.W))
    val bank        = Input(UInt(32.W))
    val group       = Input(UInt(32.W))
    val addr        = Input(UInt(32.W))
    val mask        = Input(UInt(32.W))
    val data        = Input(UInt(128.W))
  })

  addResource("/vsrc/AccessWriteDPI.sv")
}
