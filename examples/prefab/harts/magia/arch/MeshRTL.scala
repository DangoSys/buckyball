package examples.hart.magia

import chisel3.experimental.hierarchy.{instantiable, public}

import chisel3._
import chisel3.util._
import framework.generic.bus.axi4
import java.nio.file.Paths

object MeshRTL {

  class Address(bits: Int) extends Bundle {
    val id    = UInt(6.W)
    val addr  = UInt(bits.W)
    val len   = UInt(8.W)
    val size  = UInt(3.W)
    val burst = UInt(2.W)
    val lock  = Bool()
    val cache = UInt(4.W)
    val prot  = UInt(3.W)
    val qos   = UInt(4.W)
  }

}

@instantiable
class MeshRTL extends BlackBox with HasBlackBoxPath {
  override def desiredName = "MagiaMeshAXI"

  @public
  val io = IO(new Bundle {
    val clock        = Input(Clock())
    val reset        = Input(Bool())
    val boot_address = Input(UInt(32.W))
    val fetch_enable = Input(Bool())
    val aw           = Decoupled(new MeshRTL.Address(32))
    val w            = Decoupled(new axi4.WriteData(axi4.Params(32, 128, 6)))
    val b            = Flipped(Decoupled(new axi4.WriteResponse(axi4.Params(32, 128, 6))))
    val ar           = Decoupled(new MeshRTL.Address(32))
    val r            = Flipped(Decoupled(new axi4.ReadData(axi4.Params(32, 128, 6))))

  })

  addPath(Paths.get(getClass.getResource("/MagiaSources.sv").toURI).toString)
  addPath(Paths.get(getClass.getResource("/MagiaMeshAXI.stub").toURI).toString)
  addPath(Paths.get(getClass.getResource("/MagiaMeshAXI.native").toURI).toString)
  addPath(Paths.get(getClass.getResource("/hierarchy.vlt").toURI).toString)
  addPath(Paths.get(getClass.getResource("/verilator.f").toURI).toString)
  addPath(Paths.get(getClass.getResource("/hierarchy_params.v").toURI).toString)
}
