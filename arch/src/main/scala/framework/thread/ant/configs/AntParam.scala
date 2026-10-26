package framework.thread.ant.configs

import buckyball.config.AntConfig
import chisel3.util.isPow2
import framework.ip.memory.spm
import upickle.default.{macroRW, ReadWriter}

case class AntParam(
  codeBytes: Int,
  data:      spm.Params,
  shared:    spm.Params,
  taskBits:  Int) {
  require(codeBytes >= 32 && isPow2(codeBytes))
  require(data.base >= codeBytes && shared.base >= codeBytes)
  require(data.base + data.bytes <= shared.base || shared.base + shared.bytes <= data.base)
  require(data.dataBits == shared.dataBits)
  require(taskBits > 0)
}

object AntParam {

  def fromProto(p: AntConfig, shared: spm.Params): AntParam = {
    require(p.hasTls, "Ant TLS configuration is required")
    val tls = p.getTls
    AntParam(
      p.getCodeBytes,
      spm.Params(BigInt(java.lang.Long.toUnsignedString(tls.getBase)), tls.getBytes, tls.getDataBits),
      shared,
      p.getTaskBits
    )
  }

  implicit val spmRW: ReadWriter[spm.Params] = macroRW
  implicit val rw:    ReadWriter[AntParam]   = macroRW
}
