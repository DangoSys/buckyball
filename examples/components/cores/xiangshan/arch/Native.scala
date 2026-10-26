package examples.core.xiangshan

import chisel3.experimental.hierarchy.{instantiable, public}

import chisel3._
import java.nio.file.Paths
import scala.io.Source
import scala.util.Using
import chisel3.util._
import framework.ip.bus.{axi4, chi}

/** Issue B layout at the pinned upstream RTL boundary. Fields are declared LSB first. */
object Native {

  class Request extends chi.Flit {
    val qos                 = UInt(4.W)
    val tgtId               = UInt(7.W)
    val srcId               = UInt(7.W)
    val txnId               = UInt(8.W)
    val returnNid           = UInt(7.W)
    val stashNidValidEndian = Bool()
    val returnTxnId         = UInt(8.W)
    val opcode              = UInt(6.W)
    val size                = UInt(3.W)
    val addr                = UInt(48.W)
    val ns                  = Bool()
    val likelyShared        = Bool()
    val allowRetry          = Bool()
    val order               = UInt(2.W)
    val pCrdType            = UInt(4.W)
    val memAttr             = UInt(4.W)
    val snpAttr             = Bool()
    val lpid                = UInt(5.W)
    val exclSnoopMe         = Bool()
    val expCompAck          = Bool()
    val traceTag            = Bool()
    val rsvdc               = UInt(4.W)

    def fieldsLSB = Seq(
      qos,
      tgtId,
      srcId,
      txnId,
      returnNid,
      stashNidValidEndian,
      returnTxnId,
      opcode,
      size,
      addr,
      ns,
      likelyShared,
      allowRetry,
      order,
      pCrdType,
      memAttr,
      snpAttr,
      lpid,
      exclSnoopMe,
      expCompAck,
      traceTag,
      rsvdc
    )

  }

  class Response extends chi.Flit {
    val qos       = UInt(4.W)
    val tgtId     = UInt(7.W)
    val srcId     = UInt(7.W)
    val txnId     = UInt(8.W)
    val opcode    = UInt(4.W)
    val respErr   = UInt(2.W)
    val resp      = UInt(3.W)
    val fwdState  = UInt(3.W)
    val dbid      = UInt(8.W)
    val pCrdType  = UInt(4.W)
    val traceTag  = Bool()
    def fieldsLSB = Seq(qos, tgtId, srcId, txnId, opcode, respErr, resp, fwdState, dbid, pCrdType, traceTag)
  }

  class Data extends chi.Flit {
    val qos        = UInt(4.W)
    val tgtId      = UInt(7.W)
    val srcId      = UInt(7.W)
    val txnId      = UInt(8.W)
    val homeNid    = UInt(7.W)
    val opcode     = UInt(3.W)
    val respErr    = UInt(2.W)
    val resp       = UInt(3.W)
    val dataSource = UInt(3.W)
    val dbid       = UInt(8.W)
    val ccid       = UInt(2.W)
    val dataId     = UInt(2.W)
    val traceTag   = Bool()
    val rsvdc      = UInt(4.W)
    val be         = UInt(32.W)
    val data       = UInt(256.W)

    def fieldsLSB = Seq(
      qos,
      tgtId,
      srcId,
      txnId,
      homeNid,
      opcode,
      respErr,
      resp,
      dataSource,
      dbid,
      ccid,
      dataId,
      traceTag,
      rsvdc,
      be,
      data
    )

  }

  class Snoop extends chi.Flit {
    val qos         = UInt(4.W)
    val srcId       = UInt(7.W)
    val txnId       = UInt(8.W)
    val fwdNid      = UInt(7.W)
    val fwdTxnId    = UInt(8.W)
    val opcode      = UInt(5.W)
    val addr        = UInt(45.W)
    val ns          = Bool()
    val doNotGoToSd = Bool()
    val retToSrc    = Bool()
    val traceTag    = Bool()
    def fieldsLSB   = Seq(qos, srcId, txnId, fwdNid, fwdTxnId, opcode, addr, ns, doNotGoToSd, retToSrc, traceTag)
  }

  class Link extends Bundle {
    val txsactive = Output(Bool())
    val rxsactive = Input(Bool())
    val syscoreq  = Output(Bool())
    val syscoack  = Input(Bool())

    val tx = new Bundle {
      val linkactivereq = Output(Bool())
      val linkactiveack = Input(Bool())
      val req           = new chi.Channel((new Request).getWidth)
      val rsp           = new chi.Channel((new Response).getWidth)
      val dat           = new chi.Channel((new Data).getWidth)
    }

    val rx = Flipped(new Bundle {
      val linkactivereq = Output(Bool())
      val linkactiveack = Input(Bool())
      val rsp           = new chi.Channel((new Response).getWidth)
      val dat           = new chi.Channel((new Data).getWidth)
      val snp           = new chi.Channel((new Snoop).getWidth)
    })

  }

  class Commands extends Bundle {

    val request = Decoupled(new Bundle {
      val instruction = UInt(32.W)
      val pc          = UInt(64.W)
      val rs1         = UInt(64.W)
      val rs2         = UInt(64.W)

      val context = new Bundle {
        val satp      = UInt(64.W)
        val privilege = UInt(2.W)
        val sum       = Bool()
        val mxr       = Bool()
        val virt      = Bool()

        val pmp = Vec(
          64,
          new Bundle {
            val cfg  = UInt(8.W)
            val addr = UInt(46.W)
          }
        )

      }

    })

    val complete = Flipped(Valid(Bool()))
  }

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

  class Bus(addressBits: Int) extends Bundle {
    val aw = Decoupled(new Address(addressBits))
    val w  = Decoupled(new axi4.WriteData(axi4.Params(addressBits, 128, 4)))
    val b  = Flipped(Decoupled(new axi4.WriteResponse(axi4.Params(addressBits, 128, 4))))
    val ar = Decoupled(new Address(addressBits))
    val r  = Flipped(Decoupled(new axi4.ReadData(axi4.Params(addressBits, 128, 4))))
  }

}

/** Only this internal leaf knows the native RTL's pin names. */
@instantiable
class Native extends BlackBox with HasBlackBoxPath {
  override def desiredName = "XiangShanNative"

  @public
  val io = IO(new Bundle {
    val clock          = Input(Clock())
    val reset          = Input(AsyncReset())
    val io_hartId      = Input(UInt(6.W))
    val io_resetVector = Input(UInt(48.W))
    val io_nodeId      = Input(UInt(7.W))
    val io_interrupts  = Input(Vec(4, Bool()))
    val io_commands    = new Native.Commands
    val io_timeRequest = Decoupled(Bool())
    val io_time        = Flipped(Valid(UInt(64.W)))
    val io_chi         = new Native.Link
    val uncachedAXI    = Vec(1, new Native.Bus(31))
  })

  val manifest = Paths.get(getClass.getResource("/xiangshan/filelist.f").toURI)

  val files = Using.resource(Source.fromFile(manifest.toFile))(
    _.getLines().filter(_.nonEmpty).toVector
  )

  files.foreach(file => addPath(manifest.getParent.resolve(file.stripPrefix("./")).toString))

}
