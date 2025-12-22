package examples.toy.balldomain.bbus

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.instantiable
import org.chipsalliance.cde.config.Parameters
import examples.toy.balldomain.BallDomainParam
import framework.balldomain.bbus.BBus
import prototype.vector.configs.VecConfig
import prototype.matrix.configs.MatrixConfig
import prototype.im2col.configs.Im2colConfig
import prototype.transpose.configs.TransposeConfig
import prototype.relu.configs.ReluConfig
import prototype.transfer.configs.TransferConfig

/**
 * BBusModule - Ball bus module that directly extends BBus
 */
@instantiable
class BBusModule(parameter: BallDomainParam)(implicit p: Parameters)
    extends BBus(
      parameter,
      // Define Ball device generators to register
      Seq(
        () => new prototype.vector.VecBall(VecConfig.fromBallDomain(parameter), 0),
        () => new prototype.matrix.MatrixBall(MatrixConfig.fromBallDomain(parameter), 1),
        () => new prototype.im2col.Im2colBall(Im2colConfig.fromBallDomain(parameter), 2),
        () => new prototype.transpose.TransposeBall(TransposeConfig.fromBallDomain(parameter), 3),
        () => new prototype.relu.ReluBall(ReluConfig.fromBallDomain(parameter), 4),
        () =>
          new examples.toy.balldomain.emptyball.EmptyBall(
            parameter,
            5,
            parameter.bankEntries,
            parameter.bankWidth,
            parameter.bankMaskLen
          ),
        () => new prototype.transfer.TransferBall(TransferConfig.fromBallDomain(parameter), 6)
      )
    ) {
  override lazy val desiredName = "BBusModule"
}
