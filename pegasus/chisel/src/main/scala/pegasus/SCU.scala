package pegasus

import chisel3._
import chisel3.util._

// BUFGCE Xilinx primitive for glitch-free clock gating
// Must be instantiated as a BlackBox — never use Chisel when() to gate a clock
class BUFGCE extends BlackBox {
  val io = IO(new Bundle {
    val I  = Input(Clock())
    val CE = Input(Bool())
    val O  = Output(Clock())
  })
}

// AXI-Lite slave interface (from XDMA AXI-Lite master, BAR0)
// Signal direction is from the perspective of the AXI-Lite slave (SCU)
class SCUAXILiteIO extends Bundle {
  // Write address channel (Input from master)
  val awvalid = Input(Bool())
  val awready = Output(Bool())
  val awaddr  = Input(UInt(16.W))
  val awprot  = Input(UInt(3.W))

  // Write data channel
  val wvalid  = Input(Bool())
  val wready  = Output(Bool())
  val wdata   = Input(UInt(32.W))
  val wstrb   = Input(UInt(4.W))

  // Write response channel
  val bvalid  = Output(Bool())
  val bready  = Input(Bool())
  val bresp   = Output(UInt(2.W))

  // Read address channel
  val arvalid = Input(Bool())
  val arready = Output(Bool())
  val araddr  = Input(UInt(16.W))
  val arprot  = Input(UInt(3.W))

  // Read data channel
  val rvalid  = Output(Bool())
  val rready  = Input(Bool())
  val rdata   = Output(UInt(32.W))
  val rresp   = Output(UInt(2.W))
}

// Simulation Control Unit
//
// Register map (AXI-Lite, BAR0 offsets):
//   0x0000 CTRL      W [0]=freeRun, [1]=stepMode enter
//   0x0004 STEP_N    W [31:0] = number of cycles to execute; writing triggers single-step
//   0x0008 STATUS    R [0]=idle, [1]=stepDone
//   0x000C CYCLE_LO  R [31:0] = cycle counter low
//   0x0010 CYCLE_HI  R [31:0] = cycle counter high
//   0x0014 RESET     W [0] = DUT reset (1=assert, 0=deassert)
//
// Clock gating: SCU uses a BUFGCE primitive to gate the host clock.
// The SCU itself runs on the host (AXI) clock. DUT gets the gated clock.
//
class SCU extends Module {
  val io = IO(new Bundle {
    val axil      = new SCUAXILiteIO          // AXI-Lite slave (from XDMA BAR0 master)
    val host_clk  = Input(Clock())            // Host clock input (XDMA axi_aclk, 250 MHz)
    val dut_clk   = Output(Clock())           // Gated DUT clock output
    val dut_reset = Output(Bool())            // DUT reset signal (active high)
  })

  // --- Clock gate ---
  val bufgce = Module(new BUFGCE)
  bufgce.io.I := io.host_clk

  // --- Control registers (run on host clock domain, i.e. the module's implicit clock) ---
  val freeRunMode  = RegInit(false.B)
  val stepMode     = RegInit(false.B)
  val stepCount    = RegInit(0.U(32.W))
  val cycleCounter = RegInit(0.U(64.W))
  val stepDone     = RegInit(false.B)
  val dutReset     = RegInit(true.B)  // DUT starts in reset

  // Clock enable logic
  val clkEnable = Wire(Bool())
  when(stepMode) {
    clkEnable := stepCount > 0.U
    when(clkEnable) {
      stepCount    := stepCount - 1.U
      cycleCounter := cycleCounter + 1.U
    }
    // Mark step done when last cycle fires
    when(stepCount === 1.U && clkEnable) {
      stepDone := true.B
    }
  }.otherwise {
    clkEnable := freeRunMode
    when(clkEnable) {
      cycleCounter := cycleCounter + 1.U
    }
  }

  bufgce.io.CE := clkEnable
  io.dut_clk   := bufgce.io.O
  io.dut_reset := dutReset

  // --- AXI-Lite state machine ---
  // Simple two-state machine: IDLE and handle write/read
  // AW and W channels are accepted together; B response sent immediately (OKAY)
  // AR channel is accepted; R response sent with register data

  val sIdle :: sWriteResp :: sReadData :: Nil = Enum(3)
  val state = RegInit(sIdle)

  // Captured write address and data
  val wrAddr = RegInit(0.U(16.W))
  val wrData = RegInit(0.U(32.W))
  val wrStrb = RegInit(0.U(4.W))

  // Captured read address
  val rdAddr = RegInit(0.U(16.W))
  val rdData = RegInit(0.U(32.W))

  // Default outputs
  io.axil.awready := false.B
  io.axil.wready  := false.B
  io.axil.bvalid  := false.B
  io.axil.bresp   := 0.U  // OKAY
  io.axil.arready := false.B
  io.axil.rvalid  := false.B
  io.axil.rdata   := 0.U
  io.axil.rresp   := 0.U  // OKAY

  switch(state) {
    is(sIdle) {
      // Accept write if both AW and W are valid
      when(io.axil.awvalid && io.axil.wvalid) {
        io.axil.awready := true.B
        io.axil.wready  := true.B
        wrAddr          := io.axil.awaddr
        wrData          := io.axil.wdata
        wrStrb          := io.axil.wstrb
        state           := sWriteResp
      }.elsewhen(io.axil.arvalid) {
        // Accept read
        io.axil.arready := true.B
        rdAddr          := io.axil.araddr
        state           := sReadData
      }
    }
    is(sWriteResp) {
      // Apply the write and send B response
      io.axil.bvalid := true.B
      when(io.axil.bready) {
        state := sIdle
      }
      // Register writes
      switch(wrAddr(7, 0)) {
        is(0x00.U) {  // CTRL
          when(wrData(0)) {
            // CTRL[0] = 1: start free running, clear step mode
            freeRunMode := true.B
            stepMode    := false.B
          }.otherwise {
            // CTRL[0] = 0: halt
            freeRunMode := false.B
          }
          when(wrData(1)) {
            // CTRL[1] = 1: enter step mode (halts freeRun)
            stepMode    := true.B
            freeRunMode := false.B
          }
        }
        is(0x04.U) {  // STEP_N: write N → single-step N cycles
          stepCount    := wrData
          stepMode     := true.B
          freeRunMode  := false.B
          stepDone     := false.B
        }
        is(0x14.U) {  // RESET
          dutReset := wrData(0)
        }
      }
    }
    is(sReadData) {
      // Compute read data
      val rdat = Wire(UInt(32.W))
      rdat := 0.U
      switch(rdAddr(7, 0)) {
        is(0x08.U) {  // STATUS
          rdat := Cat(0.U(30.W), stepDone, (!freeRunMode && !clkEnable))
        }
        is(0x0C.U) {  // CYCLE_LO
          rdat := cycleCounter(31, 0)
        }
        is(0x10.U) {  // CYCLE_HI
          rdat := cycleCounter(63, 32)
        }
      }
      rdData := rdat

      io.axil.rvalid := true.B
      io.axil.rdata  := rdData
      when(io.axil.rready) {
        state := sIdle
      }
    }
  }
}
