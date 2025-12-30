package framework.gpdomain.configs

import upickle.default._

/**
 * GpDomain参数
 */
case class GpDomainParam(
  placeholder: Int)

object GpDomainParam {
  implicit val rw: ReadWriter[GpDomainParam] = macroRW

  /**
   * 从默认的局部JSON文件加载
   */
  def apply(): GpDomainParam = {
    val jsonStr = scala.io.Source.fromFile("src/main/scala/framework/gpdomain/configs/default.json").mkString
    read[GpDomainParam](jsonStr)
  }

}
