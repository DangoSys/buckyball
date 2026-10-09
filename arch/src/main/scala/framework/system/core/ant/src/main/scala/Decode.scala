package framework.ant

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}

object Kind extends ChiselEnum {
  val illegal, lui, auipc, jal, jalr, branch, load, store, alu, multiply, divide, custom, fence, exit, breakpoint =
    Value
}

class Decoded extends Bundle {
  val kind            = Kind()
  val rs1             = UInt(5.W)
  val rs2             = UInt(5.W)
  val rd              = UInt(5.W)
  val funct3          = UInt(3.W)
  val immediate       = UInt(64.W)
  val immediateAlu    = Bool()
  val word            = Bool()
  val subtract        = Bool()
  val arithmeticShift = Bool()
}

@instantiable
class Decode extends Module {

  @public val io = IO(new Bundle {
    val instruction = Input(UInt(32.W))
    val decoded     = Output(new Decoded)
  })

  val i  = io.instruction
  val f3 = i(14, 12)
  val f7 = i(31, 25)
  val d  = WireDefault(0.U.asTypeOf(new Decoded))
  d.kind            := Kind.illegal
  d.rs1             := i(19, 15)
  d.rs2             := i(24, 20)
  d.rd              := i(11, 7)
  d.funct3          := f3
  d.immediate       := i(31, 20).asSInt.pad(64).asUInt
  d.word            := i(3)
  d.subtract        := f7 === 32.U
  d.arithmeticShift := i(30)
  switch(i(6, 0)) {
    is("h37".U) { d.kind := Kind.lui; d.immediate := Cat(i(31, 12), 0.U(12.W)).asSInt.pad(64).asUInt }
    is("h17".U) { d.kind := Kind.auipc; d.immediate := Cat(i(31, 12), 0.U(12.W)).asSInt.pad(64).asUInt }
    is("h6f".U) {
      d.kind      := Kind.jal
      d.immediate := Cat(i(31), i(19, 12), i(20), i(30, 21), 0.U(1.W)).asSInt.pad(64).asUInt
    }
    is("h67".U)(when(f3 === 0.U)(d.kind := Kind.jalr))
    is("h63".U) {
      when(f3 === 0.U || f3 === 1.U || f3 >= 4.U)(d.kind := Kind.branch)
      d.immediate                                        := Cat(i(31), i(7), i(30, 25), i(11, 8), 0.U(1.W)).asSInt.pad(64).asUInt
    }
    is("h03".U)(when(f3 <= 6.U)(d.kind := Kind.load))
    is("h23".U) {
      when(f3 <= 3.U)(d.kind := Kind.store)
      d.immediate            := Cat(i(31, 25), i(11, 7)).asSInt.pad(64).asUInt
    }
    is("h13".U, "h1b".U) {
      d.immediateAlu := true.B
      d.subtract     := false.B
      val shiftLeft     = f3 === 1.U
      val shiftRight    = f3 === 5.U
      val shiftEncoding =
        Mux(i(3), f7 === 0.U || (shiftRight && f7 === 32.U), i(31, 26) === 0.U || (shiftRight && i(31, 26) === 16.U))
      val operation     = !i(3) || f3 === 0.U || shiftLeft || shiftRight
      when(operation && (!(shiftLeft || shiftRight) || shiftEncoding))(d.kind := Kind.alu)
    }
    is("h33".U, "h3b".U) {
      when(f7 === 1.U) {
        when(!i(3) || f3 === 0.U || f3 >= 4.U) {
          d.kind := Mux(f3 < 4.U, Kind.multiply, Kind.divide)
        }
      }.otherwise {
        val operation = !i(3) || f3 === 0.U || f3 === 1.U || f3 === 5.U
        when(operation && (f7 === 0.U || (f7 === 32.U && (f3 === 0.U || f3 === 5.U)))) {
          d.kind := Kind.alu
        }
      }
    }
    is("h7b".U)(d.kind := Kind.custom)
    is("h0f".U)(when(f3 === 0.U)(d.kind := Kind.fence))
    is("h73".U) {
      when(i === "h00000073".U)(d.kind := Kind.exit)
      when(i === "h00100073".U)(d.kind := Kind.breakpoint)
    }
  }
  io.decoded        := d
}
