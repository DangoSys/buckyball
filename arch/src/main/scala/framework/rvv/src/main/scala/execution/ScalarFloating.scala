package framework.rvv

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}

@instantiable
class ScalarFloating extends Module {

  @public
  val io = IO(new Bundle {
    val instruction  = Input(UInt(32.W))
    val source1      = Input(UInt(64.W))
    val source2      = Input(UInt(64.W))
    val source3      = Input(UInt(64.W))
    val xSource      = Input(UInt(32.W))
    val frm          = Input(UInt(3.W))
    val start        = Input(Bool())
    val ready        = Output(Bool())
    val valid        = Output(Bool())
    val legal        = Output(Bool())
    val result       = Output(UInt(64.W))
    val flags        = Output(UInt(5.W))
    val floatWrite   = Output(Bool())
    val integerWrite = Output(Bool())
  })

  val opcode         = io.instruction(6, 0)
  val funct          = io.instruction(14, 12)
  val rs2            = io.instruction(24, 20)
  val format         = io.instruction(26, 25)
  val operation      = Cat(io.instruction(31, 27), 0.U(2.W))
  val double         = format === 1.U
  val fmaInstruction = Seq(0x43, 0x47, 0x4b, 0x4f).map(n => opcode === n.U).reduce(_ || _)
  val usesRounding   =
    fmaInstruction || Seq(0, 4, 8, 12, 0x2c, 0x20, 0x60, 0x68).map(n => operation === n.U).reduce(_ || _)
  val rounding       = Mux(usesRounding, Mux(funct === 7.U, io.frm, funct), 0.U)
  val falu: Instance[FALU] = Instantiate(new FALU)
  def unbox(source: UInt): UInt = Mux(double, source, Mux(source(63, 32).andR, source(31, 0), "h7fc00000".U))
  falu.io.a            := unbox(io.source1)
  falu.io.b            := unbox(io.source2)
  falu.io.c            := unbox(io.source3)
  falu.io.sew          := Mux(double, 3.U, 2.U)
  falu.io.op           := 0.U
  falu.io.subop        := 0.U
  falu.io.roundingMode := rounding
  val decoded      = WireDefault(false.B)
  val integer      = WireDefault(false.B)
  val direct       = WireDefault(false.B)
  val directResult = WireDefault(0.U(64.W))
  when(fmaInstruction) {
    decoded    := true.B
    falu.io.op := MuxLookup(opcode, 44.U)(Seq(0x47.U -> 46.U, 0x4b.U -> 47.U, 0x4f.U -> 45.U))
  }
  when(opcode === 0x53.U) {
    switch(operation) {
      is(0.U) { decoded := true.B; falu.io.op := 0.U }
      is(4.U) { decoded := true.B; falu.io.op := 2.U }
      is(8.U) { decoded := true.B; falu.io.op := 36.U }
      is(12.U) { decoded := true.B; falu.io.op := 32.U }
      is(0x2c.U) { decoded := rs2 === 0.U; falu.io.op := 19.U; falu.io.subop := 0.U }
      is(0x10.U) { decoded := funct <= 2.U; falu.io.op := 8.U + funct }
      is(0x14.U) { decoded := funct <= 1.U; falu.io.op := Mux(funct === 0.U, 4.U, 6.U) }
      is(0x20.U) {
        decoded       := Mux(double, rs2 === 0.U, rs2 === 1.U)
        falu.io.op    := 18.U
        falu.io.sew   := 2.U
        falu.io.subop := Mux(double, 12.U, 20.U)
        falu.io.a     := Mux(double, Mux(io.source1(63, 32).andR, io.source1(31, 0), "h7fc00000".U), io.source1)
      }
      is(0x50.U) {
        decoded    := funct <= 2.U
        integer    := true.B
        falu.io.op := MuxLookup(funct, 25.U)(Seq(1.U -> 27.U, 2.U -> 24.U))
      }
      is(0x60.U) {
        decoded       := rs2 <= 1.U
        integer       := true.B
        falu.io.op    := 18.U
        falu.io.sew   := 2.U
        falu.io.subop := Mux(double, 16.U, 0.U) + Mux(rs2 === 0.U, 1.U, 0.U)
      }
      is(0x68.U) {
        decoded       := rs2 <= 1.U
        falu.io.op    := 18.U
        falu.io.sew   := 2.U
        falu.io.subop := Mux(double, 10.U, 2.U) + Mux(rs2 === 0.U, 1.U, 0.U)
        falu.io.a     := io.xSource
      }
      is(0x70.U) {
        decoded := rs2 === 0.U && (funct === 1.U || (!double && funct === 0.U))
        integer := true.B
        when(funct === 1.U) { falu.io.op := 19.U; falu.io.subop := 16.U }
          .otherwise { direct := true.B; directResult := io.source1(31, 0) }
      }
      is(0x78.U) {
        decoded      := !double && rs2 === 0.U && funct === 0.U
        direct       := true.B
        directResult := io.xSource
      }
    }
  }
  io.legal := decoded && format <= 1.U && rounding <= 4.U && (direct || falu.io.legal)
  falu.io.start := io.start && io.legal && !direct
  io.ready      := falu.io.ready
  io.valid      := Mux(direct, io.start && io.ready && io.legal, falu.io.valid)
  val result = Mux(direct, directResult, falu.io.result)
  io.result       := Mux(integer, result(31, 0), Mux(double, result, Cat("hffffffff".U(32.W), result(31, 0))))
  io.flags        := Mux(direct, 0.U, falu.io.flags)
  io.floatWrite   := io.legal && !integer
  io.integerWrite := io.legal && integer
}
