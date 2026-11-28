package framework.thread.acc.gpdomain

import buckyball.config.Design
import java.nio.file.{Files, Paths}
import scala.jdk.CollectionConverters._

object Emit extends App {
  val Array("--config", config, "--design-pb", designPath, "--output-dir", output) = args
  val placement                                                                    = ujson.read(os.read(os.Path(config)))("placement").str
  val design                                                                       = Design.parseFrom(Files.readAllBytes(Paths.get(designPath)))
  val harts                                                                        = design.getDevice.getHartsList.asScala
  val options                                                                      = Array("--target-dir", s"$output/GpDomain", "--split-verilog")
  placement match {
    case "hart"   =>
      val source = harts.find(h => h.hasAccelerator && h.getAccelerator.getGpdomain.getEnable).get
      val p      = framework.generic.acc.configs.AccParam.fromProto(source)
      _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
        new framework.custom.gpdomain.GpDomain(p.frontend, p.memDomain, p.gpDomain),
        args = options
      )
    case "thread" =>
      val (hart, thread) = harts.iterator.flatMap(h =>
        h.getThreadIndicesList.asScala.iterator.map(i => (h, design.getThreads(i.intValue)))
      ).find(_._2.getGpdomain.getEnable).get
      val p              = framework.thread.acc.configs.AccParam.fromProto(thread, hart)
      _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
        new GpDomain(p.frontend, p.memDomain, p.gpDomain),
        args = options
      )
  }
}
