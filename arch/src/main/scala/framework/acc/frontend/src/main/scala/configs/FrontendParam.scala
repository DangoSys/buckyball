package framework.acc.frontend.configs

import buckyball.config.FrontendConfig

import upickle.default._

/**
 * Frontend parameters - includes all frontend configuration.
 */
case class FrontendParam(
  ownerId:                  Int,
  commandDataBits:          Int,
  rob_entries:              Int,
  bank_alias_enable:        Boolean,
  rs_out_of_order_response: Boolean,
  bank_id_len:              Int,
  max_private_vbank:        Int,
  /** ISA bank-id namespace base for shared banks: raw id >= this means shared. */
  shared_vbank_base:        Int,
  iter_len:                 Int,
  sub_rob_enable:           Boolean,
  sub_rob_depth:            Int)

object FrontendParam {

  def fromProto(p: FrontendConfig, commandDataBits: Int, ownerId: Int): FrontendParam = {
    require(p.hasBankAliasEnable, "Frontend bank alias policy is required")
    FrontendParam(
      ownerId,
      commandDataBits,
      p.getRobEntries,
      p.getBankAliasEnable,
      p.getRsOutOfOrderResponse,
      p.getBankIdLen,
      p.getMaxPrivateVbank,
      p.getSharedVbankBase,
      p.getIterLen,
      p.getSubRobEnable,
      p.getSubRobDepth
    )
  }

  implicit val rw: ReadWriter[FrontendParam] = macroRW

}
