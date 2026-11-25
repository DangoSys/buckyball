package framework.thread.acc.configs

import chisel3.util.log2Up
import framework.generic.contract.ball.BallHostParam
import framework.generic.contract.blink.BlinkParam
import framework.thread.acc.frontend.configs.FrontendParam
import framework.thread.acc.gpdomain.configs.GpDomainParam
import framework.thread.acc.memdomain.configs.MemDomainParam

/** What this accelerator gives the Ball domain it hosts and its Balls. */
object BallLink {

  def blink(fp: FrontendParam, mp: MemDomainParam): BlinkParam = BlinkParam(
    robIdBits = log2Up(fp.rob_entries),
    dataBits = fp.commandDataBits,
    iterBits = fp.iter_len,
    bankIdBits = fp.bank_id_len,
    groupIdBits = mp.groupIdWidth,
    groupCountBits = mp.groupCountWidth,
    bankNum = mp.bankNum,
    bankEntries = mp.bankEntries,
    bankWidth = mp.bankWidth,
    bankMaskLen = mp.bankMaskLen,
    mmioBytes = mp.mmioTotalBytes,
    mmioDataBits = mp.mmioReadWidth
  )

  def host(fp: FrontendParam, mp: MemDomainParam, gp: GpDomainParam): BallHostParam = BallHostParam(
    link = blink(fp, mp),
    robEntries = fp.rob_entries,
    bankRequestPorts = mp.bankRequestPorts,
    mmioEnable = mp.mmioEnable,
    mmioBankNum = mp.mmioBankNum,
    mmioBankEntries = mp.mmioBankEntries,
    mmioBankWidth = mp.mmioBankWidth,
    subRobEnable = fp.sub_rob_enable,
    subRobDepth = fp.sub_rob_depth,
    bankAliasEnable = fp.bank_alias_enable,
    operators = gp.enable,
    memAddrBits = mp.memAddrLen
  )

}
