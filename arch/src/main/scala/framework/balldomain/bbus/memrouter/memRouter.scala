package framework.balldomain.bbus.memrouter

import chisel3._
import chisel3.util._
import examples.toy.balldomain.BallDomainParam
import framework.balldomain.rs.{BallRsComplete, BallRsIssue}
import framework.memdomain.backend.banks.{SramReadIO, SramReadReq, SramReadResp, SramWriteIO, SramWriteReq}
import framework.balldomain.blink.{BankRead, BankWrite}
import framework.balldomain.bbus.BBusConfigIO
import chisel3.experimental.hierarchy.{instantiable, public}
import chisel3.experimental.{SerializableModule, SerializableModuleParameter}

@instantiable
class MemRouter(val parameter: BallDomainParam, val numBalls: Int, val bbusChannel: Int)
    extends Module
    with SerializableModule[BallDomainParam] {

  @public
  val io = IO(new Bundle {

    val bankRead_i = Vec(
      numBalls,
      Vec(
        parameter.numBanks,
        new BankRead(parameter.bankEntries, parameter.bankWidth, parameter.rob_entries, parameter.numBanks)
      )
    )

    val bankWrite_i = Vec(
      numBalls,
      Vec(
        parameter.numBanks,
        new BankWrite(
          parameter.bankEntries,
          parameter.bankWidth,
          parameter.bankMaskLen,
          parameter.rob_entries,
          parameter.numBanks
        )
      )
    )

    val bbusConfig_i = Flipped(Decoupled(new BBusConfigIO(numBalls)))

    // Output: bbusChannel channels to MemDomain frontend
    val bankRead_o = Vec(
      bbusChannel,
      Flipped(new BankRead(parameter.bankEntries, parameter.bankWidth, parameter.rob_entries, parameter.numBanks))
    )

    val bankWrite_o = Vec(
      bbusChannel,
      Flipped(new BankWrite(
        parameter.bankEntries,
        parameter.bankWidth,
        parameter.bankMaskLen,
        parameter.rob_entries,
        parameter.numBanks
      ))
    )

  })

}
