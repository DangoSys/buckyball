package examples.balls.mxmm

import chisel3._
import chisel3.util._
import framework.memdomain.backend.banks.{SramReadReq, SramReadResp}
import framework.top.GlobalConfig

class InputWindow(b: GlobalConfig) extends Module {
  private val addressWidth = log2Ceil(b.memDomain.bankEntries)

  val io = IO(new Bundle {
    val start      = Input(Bool())
    val enable     = Input(Bool())
    val rows       = Input(UInt(12.W))
    val count      = Input(UInt(34.W))
    val fullK      = Input(UInt(16.W))
    val startK     = Input(UInt(16.W))
    val request    = Decoupled(new SramReadReq(b))
    val response   = Flipped(Decoupled(new SramReadResp(b)))
    val done       = Output(Bool())
    val write      = Output(Bool())
    val scaleWrite = Output(Bool())
    val lane       = Output(UInt(4.W))
    val address    = Output(UInt(addressWidth.W))
    val word       = Output(UInt(128.W))
    val scale      = Output(UInt(8.W))
  })

  private val request :: response :: scales :: done :: Nil = Enum(4)
  private val state                                        = RegInit(done)
  private val rows                                         = Reg(UInt(12.W))
  private val count                                        = Reg(UInt(34.W))
  private val fullK                                        = Reg(UInt(16.W))
  private val startK                                       = Reg(UInt(16.W))
  private val row                                          = Reg(UInt(12.W))
  private val offset                                       = Reg(UInt(addressWidth.W))
  private val scaleMode                                    = Reg(Bool())
  private val word                                         = Reg(UInt(128.W))
  private val scaleAddress                                 = rows * fullK + row * (fullK >> 5) + (startK >> 5) + offset
  private val byte                                         = scaleAddress(3, 0)
  io.request.valid            := io.enable && state === request
  io.request.bits.addr        := Mux(scaleMode, scaleAddress >> 4, row * (fullK >> 4) + (startK >> 4) + offset)
  io.response.ready           := io.enable && state === response
  io.done                     := state === done
  io.write                    := io.response.fire && !scaleMode
  io.scaleWrite               := io.enable && state === scales
  io.lane                     := row(3, 0)
  io.address                  := (row >> 4) * Mux(scaleMode, count >> 5, count >> 4) + offset
  io.word                     := io.response.bits.data
  io.scale                    := (word >> (byte << 3))(7, 0)
  when(io.start) {
    rows      := io.rows
    count     := io.count
    fullK     := io.fullK
    startK    := io.startK
    row       := 0.U
    offset    := 0.U
    scaleMode := false.B
    state     := request
  }
  when(io.request.fire)(state := response)
  when(io.response.fire) {
    when(scaleMode) { word := io.response.bits.data; state := scales }.otherwise {
      when(offset + 1.U === (count >> 4)) {
        offset           := 0.U
        when(row + 1.U === rows) { row := 0.U; scaleMode := true.B }
          .otherwise(row := row + 1.U)
      }.otherwise(offset := offset + 1.U)
      state              := request
    }
  }
  when(io.enable && state === scales) {
    when(offset + 1.U === (count >> 5)) {
      offset := 0.U
      when(row + 1.U === rows)(state := done)
        .otherwise { row := row + 1.U; state := request }
    }.otherwise {
      offset                    := offset + 1.U
      when(byte === 15.U)(state := request)
    }
  }
}
