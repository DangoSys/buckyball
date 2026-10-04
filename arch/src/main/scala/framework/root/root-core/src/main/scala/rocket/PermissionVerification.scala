package hier.core.rocket

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import framework.system.core.rocket.{CpuParams, HasCpuParameters}
import freechips.rocketchip.rocket.PMPConfig
import memcore.memory.cpu.PhysicalRegion
import memcore.memory.preflight.{Authorization, Permission, Params => PreparationParams}
import chisel3.util.Decoupled

@instantiable
class PermissionVerification(config: PreparationParams, regions: Seq[PhysicalRegion])(implicit val cpuParams: CpuParams)
    extends Module
    with HasCpuParameters {
  require(nPMPs == 4)

  @public val io = IO(new Bundle {
    val request      = Flipped(Decoupled(new Authorization(config)))
    val response     = Decoupled(new Permission(config))
    val contextValid = Input(Bool())
    val contextId    = Input(UInt(config.idBits.W))
    val pmpConfig    = Input(UInt(32.W))
    // PMP address CSRs expressed as CSR << 2, including their NAPOT encoding.
    val pmpAddresses = Input(Vec(4, UInt(64.W)))
  })

  val permission = Instantiate(new PermissionCheck(config, regions))
  permission.io.request <> io.request
  io.response <> permission.io.response
  permission.io.contextValid := io.contextValid
  permission.io.contextId    := io.contextId
  for (i <- 0 until 4) {
    val pmp = permission.io.pmp(i)
    pmp.cfg  := io.pmpConfig(8 * i + 7, 8 * i).asTypeOf(new PMPConfig)
    pmp.addr := io.pmpAddresses(i)(paddrBits - 1, 2)
    pmp.mask := pmp.computeMask
  }
}
