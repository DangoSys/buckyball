package framework.frontend.configs

import upickle.default._

/**
 * Frontend Parameter - 包含前端所有配置
 */
case class FrontendParam(
  rob_entries:              Int,
  rs_out_of_order_response: Boolean,
  bank_id_len:              Int,
  iter_len:                 Int,
  sub_rob_depth:            Int)

object FrontendParam {
  implicit val rw: ReadWriter[FrontendParam] = macroRW

  def apply(): FrontendParam = {
    val jsonStr = scala.io.Source.fromFile("src/main/scala/framework/frontend/configs/default.json").mkString
    read[FrontendParam](jsonStr)
  }

}
