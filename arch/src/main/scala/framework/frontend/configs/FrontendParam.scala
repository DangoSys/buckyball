package framework.frontend.configs

import upickle.default._

/**
 * Frontend parameters - includes all frontend configuration.
 */
case class FrontendParam(
  rob_entries:              Int,
  rs_out_of_order_response: Boolean,
  bank_id_len:              Int,
  vbank_id_upper_bound:     Int,
  /** ISA bank-id namespace base for shared banks: raw id >= this means shared. */
  shared_bank_id_base:      Int,
  iter_len:                 Int,
  sub_rob_enable:           Boolean,
  sub_rob_depth:            Int)

object FrontendParam {
  implicit val rw: ReadWriter[FrontendParam] = macroRW

  def apply(): FrontendParam = FrontendParam(
    rob_entries = 0,
    rs_out_of_order_response = false,
    bank_id_len = 0,
    vbank_id_upper_bound = 0,
    shared_bank_id_base = 0,
    iter_len = 0,
    sub_rob_enable = false,
    sub_rob_depth = 0
  )

}
