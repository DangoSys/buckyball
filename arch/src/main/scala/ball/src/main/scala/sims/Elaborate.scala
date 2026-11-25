package framework.ball.sims

import _root_.circt.stage.ChiselStage
import chisel3.util.log2Up
import framework.ball.BallDomain
import framework.ball.bbus.BallFactory
import framework.generic.contract.ball.BallHostParam
import framework.generic.contract.blink.BlinkParam
import framework.generic.contract.configs.BallDomainParam
import framework.generic.contract.memory.access.preflight.configs.PmpParams

/**
 * A Ball alone: a BallDomain that holds only this Ball, in the host of design.pb's `ball`. The Ball's factory is its
 * RTL class, so the entry needs no table of the Balls there are.
 */
object Elaborate {

  def host(h: buckyball.config.BallHost): BallHostParam = {
    // One private bank space: the group widths of the memory domain without shared banks.
    val link = BlinkParam(
      robIdBits = log2Up(h.getRobEntries),
      dataBits = h.getDataBits,
      iterBits = h.getIterBits,
      bankIdBits = h.getBankIdBits,
      groupIdBits = log2Up(h.getBankNum),
      groupCountBits = log2Up(h.getBankNum + 1),
      bankNum = h.getBankNum,
      bankEntries = h.getBankEntries,
      bankWidth = h.getBankWidth,
      bankMaskLen = h.getBankMaskLen,
      mmioBytes = h.getMmioBankNum * h.getMmioBankEntries * (h.getMmioBankWidth / 8),
      mmioDataBits = h.getMmioReadWidth
    )
    BallHostParam(
      link = link,
      robEntries = h.getRobEntries,
      bankRequestPorts = h.getBankRequestPorts,
      mmioEnable = h.getMmioEnable,
      mmioBankNum = h.getMmioBankNum,
      mmioBankEntries = h.getMmioBankEntries,
      mmioBankWidth = h.getMmioBankWidth,
      subRobEnable = h.getSubRobEnable,
      subRobDepth = h.getSubRobDepth,
      bankAliasEnable = h.getBankAliasEnable,
      operators = h.getOperators,
      memAddrBits = h.getMemAddrBits
    )
  }

  def main(args: Array[String]): Unit = {
    require(args.length >= 6 && args(0) == "--config" && args(2) == "--target-class" && args(4) == "--target-dir")
    val design     = buckyball.config.Design.parseFrom(java.nio.file.Files.readAllBytes(java.nio.file.Paths.get(args(1))))
    require(design.hasBall, s"${args(1)}: not the design.pb of a Ball alone")
    val ball       = design.getBall
    val domain     = BallDomainParam.fromProto(ball.getBalldomain)
    val Seq(route) = domain.balls
    val factory    = Class.forName(route.rtlClass + "$").getField("MODULE$").get(null).asInstanceOf[BallFactory]
    val pmp        = PmpParams(ball.getHost.getPmpAddressBits, ball.getHost.getPmpEntries)
    ChiselStage.emitSystemVerilogFile(
      new BallDomain(host(ball.getHost), domain, pmp, Map(route.component -> factory)),
      args = Array("--target-dir", args(5), "--split-verilog"),
      firtoolOpts = args.drop(6)
    )
  }

}
