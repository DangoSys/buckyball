package examples.toy.balldomain.bbus

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.instantiable
import org.chipsalliance.cde.config.Parameters
import framework.top.GlobalConfig
import framework.balldomain.bbus.BBus
import framework.balldomain.prototype.vector.configs.VectorBallParam
// import prototype.matrix.configs.MatrixConfig
// import prototype.im2col.configs.Im2colConfig
// import prototype.transpose.configs.TransposeConfig
// import prototype.relu.configs.ReluConfig
// import prototype.transfer.configs.TransferConfig

/**
 * BBusModule - Ball bus module that directly extends BBus
 */
@instantiable
class BBusModule(b: GlobalConfig)
    extends BBus(
      b,
      // Define Ball device generators to register
      Seq(
        () => new framework.balldomain.prototype.vector.VecBall(b, 0),
        // () => new prototype.matrix.MatrixBall(MatrixConfig.fromBallDomain(parameter), 1),
        // () => new prototype.im2col.Im2colBall(Im2colConfig.fromBallDomain(parameter), 2),
        // () => new prototype.transpose.TransposeBall(TransposeConfig.fromBallDomain(parameter), 3),
        // () => new prototype.relu.ReluBall(ReluConfig.fromBallDomain(parameter), 4),
        () => new examples.toy.balldomain.emptyball.EmptyBall(b, 5)
        // () => new prototype.transfer.TransferBall(TransferConfig.fromBallDomain(parameter), 6)
      )
    ) {}
