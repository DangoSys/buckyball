package hier.core.rocket

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.system.core.rocket.{CpuParams, HasCpuParameters}
import freechips.rocketchip.rocket.{PMP, PMPChecker}
import memcore.memory.cpu.PhysicalRegion
import memcore.memory.preflight.{Authorization, Permission, Params => PreparationParams}

/** Checks the complete transport range against the submitting command's frozen PMP image. */
@instantiable
class PermissionCheck(config: PreparationParams, regions: Seq[PhysicalRegion])(implicit val cpuParams: CpuParams)
    extends Module
    with HasCpuParameters {
  private val physicalBits = math.min(config.bus.addressBits, paddrBits)
  require(regions.nonEmpty)
  regions.foreach(r => require(r.base >= 0 && r.bytes > 0 && r.base + r.bytes <= (BigInt(1) << physicalBits)))

  @public val io = IO(new Bundle {
    val request      = Flipped(Decoupled(new Authorization(config)))
    // Selected from the captured admission context using request.bits.id, never the live CPU CSRs.
    val contextValid = Input(Bool())
    val contextId    = Input(UInt(config.idBits.W))
    val pmp          = Input(Vec(nPMPs, new PMP))
    val response     = Decoupled(new Permission(config))
  })

  val idle :: checking :: reporting :: Nil = Enum(3)
  val state                                = RegInit(idle)
  val request                              = Reg(new Authorization(config))
  val capturedPmp                          = Reg(Vec(nPMPs, new PMP))
  val address                              = Reg(UInt(config.bus.addressBits.W))
  val remaining                            = Reg(UInt(13.W))
  val allowed                              = Reg(Bool())
  io.request.ready       := state === idle && !reset.asBool
  io.response.valid      := state === reporting && !reset.asBool
  io.response.bits.id    := request.id
  io.response.bits.allow := allowed

  when(io.request.fire) {
    val r             = io.request.bits
    val end           = r.pa +& r.bytes
    val shape         = io.contextValid && io.contextId === r.id && r.bytes =/= 0.U && r.bytes <= 4096.U &&
      end <= (BigInt(1) << physicalBits).U &&
      (r.privilege === 0.U || r.privilege === 1.U || r.privilege === 3.U) &&
      (!r.isPte || (!r.write && r.bytes === 8.U && r.privilege === 1.U))
    val regionAllowed = regions.map { region =>
      val permission = Mux(r.write, region.writable.B, region.readable.B)
      region.normal.B && r.pa >= region.base.U && end <= (region.base + region.bytes).U && permission
    }.reduce(_ || _)
    request := r
    capturedPmp := io.pmp
    address     := r.pa
    remaining   := r.bytes
    allowed     := shape && regionAllowed
    state       := Mux(shape && regionAllowed, checking, reporting)
  }

  // A page segment may cover different PMP entries. Check each actual transport
  // transaction (at most one 64-byte line), never a widened page-sized access.
  val size = (1 to 6).foldLeft(0.U(3.W)) { (selected, logBytes) =>
    Mux(remaining >= (1 << logBytes).U && address(logBytes - 1, 0) === 0.U, logBytes.U, selected)
  }

  val bytes   = (1.U(13.W) << size)(12, 0)
  val checker = Module(new PMPChecker(6))
  checker.io.pmp  := capturedPmp
  checker.io.prv  := request.privilege
  checker.io.addr := address.pad(paddrBits)(paddrBits - 1, 0)
  checker.io.size := size
  val permitted = Mux(request.write, checker.io.w, checker.io.r)
  when(state === checking) {
    when(!permitted) {
      allowed := false.B
      state   := reporting
    }.elsewhen(remaining === bytes) {
      state := reporting
    }.otherwise {
      address   := address + bytes
      remaining := remaining - bytes
    }
  }
  when(io.response.fire)(state := idle)
}
