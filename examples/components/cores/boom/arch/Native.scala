package examples.core.boom

import chisel3.experimental.hierarchy.{instantiable, public}

import chisel3._
import chisel3.util._
import framework.ip.bus.axi4
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
  override def desiredName = "BoomNative"

  @public
  val io = IO(new Bundle {
    val clock          = Input(Clock())
    val reset          = Input(Bool())
    val io_resetVector = Input(UInt(44.W))
    val io_interrupts  = Input(Vec(4, Bool()))

    val io_commands = new Bundle {

      val request = Decoupled(new Bundle {
        val instruction = UInt(32.W)
        val rs1         = UInt(64.W)
        val rs2         = UInt(64.W)
        val privilege   = UInt(2.W)
        val sum         = Bool()
        val mxr         = Bool()
      })

      val busy  = Input(Bool())
      val fault = Input(Bool())
    }

    val io_context = Output(new Bundle {
      val satp = UInt(64.W)
      val pmp  = Vec(8, new Bundle { val cfg = UInt(8.W); val addr = UInt(42.W) })
    })

    val memoryAXI      = Vec(1, new Native.Bus(44))
    val peripheralsAXI = Vec(1, new Native.Bus(31))
    val dmaAXI         = Flipped(Vec(1, new Native.Bus(44)))
  })

  val manifest = Paths.get(getClass.getResource("/boom/filelist.f").toURI)
  val files    = Using.resource(Source.fromFile(manifest.toFile))(_.getLines().filter(_.nonEmpty).toVector)
  files.foreach(file => addPath(manifest.getParent.resolve(file.stripPrefix("./")).toString))
}
