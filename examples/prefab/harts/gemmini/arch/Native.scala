package examples.hart.gemmini

import chisel3.experimental.hierarchy.{instantiable, public}

import chisel3._
import chisel3.util._
import framework.generic.bus.axi4
import java.nio.file.Paths
import scala.io.Source
import scala.util.Using

object Native {

  class Address(bits: Int) extends Bundle {
    val id    = UInt(4.W)
    val addr  = UInt(bits.W)
    val len   = UInt(8.W)
    val size  = UInt(3.W)
    val burst = UInt(2.W)
    val lock  = Bool()
    val cache = UInt(4.W)
    val prot  = UInt(3.W)
    val qos   = UInt(4.W)
  }

  class Bus(bits: Int) extends Bundle {
    val aw = Decoupled(new Address(bits))
    val w  = Decoupled(new axi4.WriteData(axi4.Params(bits, 128, 4)))
    val b  = Flipped(Decoupled(new axi4.WriteResponse(axi4.Params(bits, 128, 4))))
    val ar = Decoupled(new Address(bits))
    val r  = Flipped(Decoupled(new axi4.ReadData(axi4.Params(bits, 128, 4))))
  }

}

@instantiable
class Native extends BlackBox with HasBlackBoxPath {
  override def desiredName = "GemminiHartNative"

  @public
  val io = IO(new Bundle {
    val clock          = Input(Clock())
    val reset          = Input(Bool())
    val io_resetVector = Input(UInt(44.W))
    val io_interrupts  = Input(Vec(4, Bool()))

    val memoryAXI      = Vec(1, new Native.Bus(44))
    val peripheralsAXI = Vec(1, new Native.Bus(31))
  })

  val manifest = Paths.get(getClass.getResource("/gemmini/filelist.f").toURI)
  val files    = Using.resource(Source.fromFile(manifest.toFile))(_.getLines().filter(_.nonEmpty).toVector)
  files.foreach(file => addPath(manifest.getParent.resolve(file.stripPrefix("./")).toString))
}
