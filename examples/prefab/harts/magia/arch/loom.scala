package examples.hart.magia

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import examples.hart.rocket.Execution
import examples.hart.rocket.configs.{Params => RocketParam}
import examples.hart.rocket.pipeline.CpuParams
import framework.hart.configs.HartParam
import framework.mlp.fetch.configs.{Params => FetchParam}
import framework.hart.memory.{MemorySystem}
import framework.generic.bus.{axi4, chi}
import freechips.rocketchip.rocket.{DCacheParams, ICacheParams, MulDivParams, RocketCoreParams}
import freechips.rocketchip.tile.FPUParams
import framework.generic.contract.hlink.HartInterrupts

/** MAGIA's process CPU and mesh scheduling are fixed parts of this hart. */
@instantiable
class Loom(p: HartParam, fetch: FetchParam) extends Module {

  @public val io = IO(new Bundle {
    val interrupts = Input(new HartInterrupts)
    val coherent   = Vec(2, new chi.RequesterPort(p.hlink.line))
    val uncached   = new axi4.Port(p.hlink.uncached)
    val boot       = Output(UInt(32.W))
    val windows    = Output(Vec(3, UInt(p.hlink.ddr.addressBits.W)))
    val running    = Output(Bool())
    val completed  = Input(Bool())
    val result     = Input(UInt(32.W))
    val output     = Flipped(Decoupled(UInt(8.W)))
  })

  val config = RocketParam(
    CpuParams(
      RocketCoreParams(
        xLen = 64,
        pgLevels = 3,
        nPMPs = 16,
        useVM = true,
        useSupervisor = true,
        useUser = true,
        useHypervisor = false,
        useDebug = false,
        useAtomics = true,
        useCompressed = true,
        nLocalInterrupts = 1,
        clockGate = false,
        mulDiv = Some(MulDivParams()),
        // Half precision too: Loom converts the operands and results of the mesh, which are FP16.
        fpu = Some(FPUParams(minFLen = 16))
      ),
      Some(DCacheParams()),
      Some(ICacheParams()),
      None,
      36,
      1
    ),
    p.memory,
    fetch,
    p.hlink.copy(requesterIds = p.hlink.requesterIds.take(2)),
    p.privileged.permission,
    p.hartId,
    p.resetVector,
    p.mtimeAddress,
    None
  )

  val execution = Instantiate(new Execution(config, commandsEnabled = true))
  val memory    = Instantiate(new MemorySystem(p.memory, externalHome = true))
  execution.io.interrupts              := io.interrupts
  memory.io.fetch <> execution.io.fetch
  memory.io.memory <> execution.io.data
  memory.io.probe.get <> execution.io.probe
  memory.io.timeRequest <> execution.io.timeRequest
  execution.io.timeResponse <> memory.io.timeResponse
  memory.io.commandAllowed             := true.B
  io.coherent <> memory.io.coherent
  io.uncached <> memory.io.uncached
  memory.io.services.cpuAllow          := true.B
  memory.io.services.maintenance.valid := false.B
  memory.io.services.maintenance.bits  := 0.U.asTypeOf(memory.io.services.maintenance.bits)
  memory.io.services.pteRequest.valid  := false.B
  memory.io.services.pteRequest.bits   := 0.U.asTypeOf(memory.io.services.pteRequest.bits)
  memory.io.services.maintained.ready  := true.B
  memory.io.services.lineAddress       := 0.U
  memory.io.services.lineWrite         := false.B
  memory.io.services.pteResponse.ready := true.B
  execution.io.fault.fault.valid       := false.B
  execution.io.fault.fault.bits        := 0.U.asTypeOf(execution.io.fault.fault.bits)

  val running = RegInit(false.B)
  val done    = RegInit(false.B)
  val result  = Reg(UInt(32.W))
  val boot    = Reg(UInt(32.W))
  io.running := running
  io.boot    := boot
  when(io.completed) {
    running := false.B
    done    := true.B
    result  := io.result
  }

  // custom-0: funct7=0 launches the physical entry in rs1; 1 reads status; 2 drains mesh output;
  // 3 points window rs2 of the mesh's view of the DDR (W, X, Z) at the page in rs1.
  // Status bit 0 is completion; bits 63:32 carry the mesh's aggregate exit code.
  val rocc        = execution.io.rocc
  val instruction = rocc.cmd.bits.instruction.raw_inst
  val launch      = instruction(31, 25) === 0.U
  val print       = instruction(31, 25) === 2.U
  val window      = instruction(31, 25) === 3.U
  // Until Loom moves them, the windows show the DDR at their own addresses.
  val windows     = RegInit(VecInit(Seq("hd0000000", "hf0000000", "hf8000000").map(_.U(p.hlink.ddr.addressBits.W))))
  io.windows := windows
  val pending  = RegInit(false.B)
  val response = Reg(chiselTypeOf(rocc.resp.bits))
  rocc.cmd.ready               := !pending && (!(launch || window) || !running)
  rocc.resp.valid              := pending
  rocc.resp.bits               := response
  rocc.busy                    := pending
  rocc.interrupt               := false.B
  io.output.ready              := rocc.cmd.fire && print
  when(rocc.resp.fire)(pending := false.B)
  when(rocc.cmd.fire) {
    assert(instruction(6, 0) === "h0b".U && instruction(31, 25) <= 3.U, "Invalid MAGIA mesh command")
    response.rd   := instruction(11, 7)
    response.data := Mux(
      print,
      Cat(io.output.valid, Mux(io.output.valid, io.output.bits, 0.U)),
      Mux(launch, 0.U, Cat(Mux(done, result, 0.U), 0.U(31.W), done))
    )
    pending       := instruction(14)
    when(launch) {
      assert(
        rocc.cmd.bits.instruction.rs1Data(63, 32) === 0.U &&
          rocc.cmd.bits.instruction.rs1Data(1, 0) === 0.U,
        "Invalid MAGIA mesh entry"
      )
      boot    := rocc.cmd.bits.instruction.rs1Data(31, 0)
      running := true.B
      done    := false.B
    }
    when(window) {
      assert(
        rocc.cmd.bits.instruction.rs2Data < 3.U &&
          rocc.cmd.bits.instruction.rs1Data(63, p.hlink.ddr.addressBits) === 0.U &&
          rocc.cmd.bits.instruction.rs1Data(11, 0) === 0.U,
        "Invalid MAGIA mesh window"
      )
      windows(rocc.cmd.bits.instruction.rs2Data(1, 0)) :=
        rocc.cmd.bits.instruction.rs1Data(p.hlink.ddr.addressBits - 1, 0)
    }
  }
}
