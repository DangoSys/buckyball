package framework.generic.bus.axi4

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}

/** Splits aligned whole-beat INCR bursts while preserving concurrent AXI IDs. */
@instantiable
class Narrow(wide: Params, narrow: Params) extends Module {
  require(wide.addressBits == narrow.addressBits && wide.idBits == narrow.idBits)
  require(wide.dataBits > narrow.dataBits && wide.dataBits % narrow.dataBits == 0)
  val rows    = wide.dataBits / narrow.dataBits
  val rowBits = log2Ceil(rows)

  @public
  val io = IO(new Bundle {
    val in  = Flipped(new Port(wide))
    val out = new Port(narrow)
  })

  val writeRow = RegInit(0.U(rowBits.W))
  io.out.aw.valid              := io.in.aw.valid
  io.in.aw.ready               := io.out.aw.ready
  io.out.aw.bits               := io.in.aw.bits
  io.out.aw.bits.len           := Cat(io.in.aw.bits.len(7 - rowBits, 0), Fill(rowBits, 1.U(1.W)))
  io.out.aw.bits.size          := log2Ceil(narrow.bytes).U
  when(io.out.aw.fire) {
    assert(io.in.aw.bits.burst === 1.U && io.in.aw.bits.size === log2Ceil(wide.bytes).U &&
      io.in.aw.bits.addr(log2Ceil(wide.bytes) - 1, 0) === 0.U && io.in.aw.bits.len < (256 / rows).U)
  }
  io.out.w.valid               := io.in.w.valid
  io.out.w.bits.data           := (io.in.w.bits.data >> (writeRow * narrow.dataBits.U))(narrow.dataBits - 1, 0)
  io.out.w.bits.strb           := (io.in.w.bits.strb >> (writeRow * narrow.bytes.U))(narrow.bytes - 1, 0)
  io.out.w.bits.last           := io.in.w.bits.last && writeRow === (rows - 1).U
  io.in.w.ready                := io.out.w.ready && writeRow === (rows - 1).U
  when(io.out.w.fire)(writeRow := writeRow + 1.U)
  io.in.b <> io.out.b

  val ids          = 1 << wide.idBits
  val readRow      = RegInit(VecInit(Seq.fill(ids)(0.U(rowBits.W))))
  val readData     = Reg(Vec(ids, Vec(rows - 1, UInt(narrow.dataBits.W))))
  val readResponse = RegInit(VecInit(Seq.fill(ids)(0.U(2.W))))
  val readOutput   = Reg(new ReadData(wide))
  val readValid    = RegInit(false.B)
  val rid          = io.out.r.bits.id
  val row          = readRow(rid)
  io.out.ar.valid     := io.in.ar.valid
  io.in.ar.ready      := io.out.ar.ready
  io.out.ar.bits      := io.in.ar.bits
  io.out.ar.bits.len  := Cat(io.in.ar.bits.len(7 - rowBits, 0), Fill(rowBits, 1.U(1.W)))
  io.out.ar.bits.size := log2Ceil(narrow.bytes).U
  when(io.out.ar.fire) {
    assert(io.in.ar.bits.burst === 1.U && io.in.ar.bits.size === log2Ceil(wide.bytes).U &&
      io.in.ar.bits.addr(log2Ceil(wide.bytes) - 1, 0) === 0.U && io.in.ar.bits.len < (256 / rows).U)
  }
  io.in.r.valid       := readValid
  io.in.r.bits        := readOutput
  io.out.r.ready      := row =/= (rows - 1).U || !readValid || io.in.r.ready
  when(io.in.r.fire) {
    readValid := false.B
  }
  when(io.out.r.fire) {
    for (id <- 0 until ids) {
      when(rid === id.U) {
        for (part <- 0 until rows - 1) {
          when(readRow(id) === part.U)(readData(id)(part) := io.out.r.bits.data)
        }
        readResponse(id) := Mux(readRow(id) === (rows - 1).U, 0.U, readResponse(id) | io.out.r.bits.resp)
        readRow(id) := readRow(id) + 1.U
      }
    }
    assert(!io.out.r.bits.last || row === (rows - 1).U)
    when(row === (rows - 1).U) {
      readOutput.data := Cat(io.out.r.bits.data, readData(rid).asUInt)
      readOutput.id   := io.out.r.bits.id
      readOutput.resp := readResponse(rid) | io.out.r.bits.resp
      readOutput.last := io.out.r.bits.last
      readValid       := true.B
    }
  }
}
