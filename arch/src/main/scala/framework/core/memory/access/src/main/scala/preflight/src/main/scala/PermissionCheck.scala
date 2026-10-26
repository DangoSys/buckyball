package framework.core.memory.access.preflight

import framework.core.memory.access.preflight.configs.PmpParams

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.core.memory.access.preflight.configs.PmpParams
import framework.core.memory.access.preflight.PmpEntry
import framework.core.memory.access.PhysicalRegion
import framework.core.memory.access.preflight.{Authorization, Permission, Params => PreparationParams}

/** Checks the complete transport range against the submitting command's frozen PMP image. */
@instantiable
class PermissionCheck(val specification: configs.Config) extends Module {
  val config              = specification.parameters
  val regions             = specification.regions
  val permissionParam     = specification.permission
  implicit val permission = permissionParam

  val physicalBits = math.min(config.bus.addressBits, permission.addressBits)
  require(regions.nonEmpty)
  regions.foreach(r => require(r.base >= 0 && r.bytes > 0 && r.base + r.bytes <= (BigInt(1) << physicalBits)))

  @public
  val io = IO(new Bundle {
    val request      = Flipped(Decoupled(new Authorization(config)))
    // Selected from the captured admission context using request.bits.id, never the live CPU CSRs.
    val contextValid = Input(Bool())
    val contextId    = Input(UInt(config.idBits.W))
    val pmp          = Input(Vec(permission.entries, new PmpEntry(permissionParam)))
    val response     = Decoupled(new Permission(config))
  })

  val idle :: checking :: reporting :: Nil = Enum(3)
  val state                                = RegInit(idle)
  val request                              = Reg(new Authorization(config))
  val capturedPmp                          = Reg(Vec(permission.entries, new PmpEntry(permissionParam)))
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
      val accessAllowed = Mux(r.write, region.writable.B, region.readable.B)
      region.normal.B && r.pa >= region.base.U && end <= (region.base + region.bytes).U && accessAllowed
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

  val bytes     = (1.U(13.W) << size)(12, 0)
  val accessEnd = address +& bytes
  val machine   = request.privilege === 3.U

  val permitted =
    (0 until permission.entries).reverse.foldLeft(if (permission.entries == 0) true.B else machine) { (previous, i) =>
      val entry         = capturedPmp(i)
      val encoded       = (entry.addr << 2).pad(permission.addressBits)
      val low           = Mux(
        entry.cfg.a === 1.U,
        if (i == 0) 0.U(permission.addressBits.W) else (capturedPmp(i - 1).addr << 2),
        Mux(entry.cfg.a === 3.U, encoded & ~entry.mask, encoded)
      )
      val high          = Mux(
        entry.cfg.a === 1.U,
        encoded.pad(permission.addressBits + 1),
        Mux(entry.cfg.a === 3.U, low +& entry.mask + 1.U, encoded +& 4.U)
      )
      val hit           = entry.cfg.a =/= 0.U && low < high && address < high && accessEnd > low
      val contained     = address >= low && accessEnd <= high
      val accessAllowed = Mux(request.write, entry.cfg.w, entry.cfg.r) || (machine && !entry.cfg.l)
      Mux(hit, contained && accessAllowed, previous)
    }

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
