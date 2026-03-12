package pegasus

import chisel3._
import chisel3.util._

// UARTCapture: receives UART TX from DUT, decodes bytes, buffers in FIFO,
// and outputs via AXI-Stream to the XDMA C2H channel.
//
// UART frame format: 1 start bit (0), 8 data bits (LSB first), 1 stop bit (1)
// Baud rate: 115200 baud by default
// Clock: host_clk (250 MHz from XDMA)
//
// The DUT's UART TX is already a standard async serial signal (the DUT's
// UART peripheral encodes the bytes into UART frames internally).
// This module acts as a UART receiver (RX) that samples the DUT's TX line.
//
// Parameters:
//   clockFreqHz : host clock frequency in Hz (default 250 MHz)
//   baudRate    : UART baud rate (default 115200)
//   fifoDepth   : byte FIFO depth (default 4096)
//
class UARTCapture(
  clockFreqHz: Int = 250_000_000,
  baudRate:    Int = 115200,
  fifoDepth:   Int = 4096
) extends Module {
  val io = IO(new Bundle {
    val uart_tx   = Input(Bool())    // DUT UART TX line (idle high)

    // AXI-Stream master output → XDMA C2H channel
    val axis_tvalid = Output(Bool())
    val axis_tready = Input(Bool())
    val axis_tdata  = Output(UInt(8.W))
    val axis_tlast  = Output(Bool())
    val axis_tkeep  = Output(UInt(1.W))
  })

  // --- Baud rate configuration ---
  // Number of clock cycles per UART bit
  val cyclesPerBit   = (clockFreqHz / baudRate).U
  // Half period offset for sampling in middle of bit
  val halfCyclesPerBit = (clockFreqHz / baudRate / 2).U

  // --- Input synchronizer (2-FF for metastability) ---
  val rxd_sync0 = RegNext(io.uart_tx, true.B)
  val rxd_sync1 = RegNext(rxd_sync0, true.B)
  val rxd       = rxd_sync1

  // --- UART RX state machine ---
  val sIdle :: sStart :: sData :: sStop :: Nil = Enum(4)
  val state = RegInit(sIdle)

  val bitCounter  = RegInit(0.U(4.W))      // counts 0..7 data bits
  val clockCount  = RegInit(0.U(32.W))     // counts cycles within a bit period
  val shiftReg    = RegInit(0.U(8.W))      // shift register for received byte
  val byteValid   = RegInit(false.B)       // pulse: one byte received
  val receivedByte = RegInit(0.U(8.W))

  byteValid := false.B  // default

  switch(state) {
    is(sIdle) {
      // Wait for start bit (falling edge: idle is 1, start bit is 0)
      when(!rxd) {
        state      := sStart
        clockCount := 0.U
      }
    }
    is(sStart) {
      // Wait half a bit period, then verify start bit is still low
      clockCount := clockCount + 1.U
      when(clockCount >= halfCyclesPerBit - 1.U) {
        when(!rxd) {
          // Valid start bit; begin receiving data bits
          state      := sData
          clockCount := 0.U
          bitCounter := 0.U
          shiftReg   := 0.U
        }.otherwise {
          // Spurious glitch, return to idle
          state := sIdle
        }
      }
    }
    is(sData) {
      // Sample each data bit at center of bit period
      clockCount := clockCount + 1.U
      when(clockCount >= cyclesPerBit - 1.U) {
        clockCount := 0.U
        // Sample bit (LSB first)
        shiftReg   := Cat(rxd, shiftReg(7, 1))
        bitCounter := bitCounter + 1.U
        when(bitCounter === 7.U) {
          state := sStop
        }
      }
    }
    is(sStop) {
      // Wait for stop bit
      clockCount := clockCount + 1.U
      when(clockCount >= cyclesPerBit - 1.U) {
        when(rxd) {
          // Valid stop bit: byte received
          byteValid    := true.B
          receivedByte := shiftReg
        }
        // Return to idle regardless (even on framing errors, resync)
        state      := sIdle
        clockCount := 0.U
      }
    }
  }

  // --- Byte FIFO ---
  val fifo = Module(new Queue(UInt(8.W), fifoDepth))
  fifo.io.enq.valid := byteValid
  fifo.io.enq.bits  := receivedByte

  // AXI-Stream output from FIFO
  io.axis_tvalid := fifo.io.deq.valid
  io.axis_tdata  := fifo.io.deq.bits
  io.axis_tlast  := false.B   // No natural frame boundary for UART bytes
  io.axis_tkeep  := 1.U       // 1 valid byte per beat
  fifo.io.deq.ready := io.axis_tready

  // Drop silently if FIFO is full (overflow protection)
  // The FIFO's enq.ready going low just means we lose bytes
}
