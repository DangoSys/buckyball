package framework.hart.cbus

import buckyball.config.Design
import chisel3._
import framework.generic.bus.chi
import framework.generic.ddr.{Params => DdrParams}
import framework.hart.configs.HartParam
import framework.hart.hlink.HLinkIO
import framework.hart.hlink.configs.{Params => HLinkParams}
import java.nio.file.{Files, Paths}

object Emit extends App {
  val Array("--config", config, "--design-pb", designPath, "--output-dir", output) = args
  val v                                                                            = ujson.read(os.read(os.Path(config)))
  val design                                                                       = Design.parseFrom(Files.readAllBytes(Paths.get(designPath)))
  require(design.hasDevice && design.getDevice.getHartsCount > 0, "CBus verification requires a device")
  val source                                                                       = design.getDevice.getHarts(0)
  val hart                                                                         = HartParam.fromProto(design, source, externalHome = false)

  val ddr = DdrParams(
    line = chi.Params(addressBits = v("addressBits").num.toInt),
    dataBits = v("hartDataBits").num.toInt,
    idBits = v("idBits").num.toInt,
    beatRows = v("beatRows").num.toInt
  )

  // The two HLink transactors can independently issue the same AXI ID.
  val link  = HLinkParams(1, false, Seq.empty, ddr.line, ddr.axi, ddr.port, false, false)
  val harts = (0 until 2).map(i => hart.copy(hartId = i, hlink = link))
  val peers = harts.indices.map(i => () => IO(Flipped(new HLinkIO(link))).suggestName(s"peer$i"))
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new CBus(harts, ddr, peers),
    args = Array("--target-dir", s"$output/CBus", "--split-verilog")
  )
}
