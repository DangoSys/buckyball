package examples.toy.balldomain.bbus

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.instantiable
import framework.top.GlobalConfig
import framework.balldomain.bbus.BBus
import framework.balldomain.blink.BallRegist
import framework.balldomain.prototype.vector.VecBall
import framework.balldomain.prototype.relu.ReluBall

/**
 * BBusModule - Ball bus module that directly extends BBus
 */
@instantiable
class BBusModule(b: GlobalConfig)
    extends BBus(
      b,
      b.ballDomain.ballIdMappings.map { mapping =>
        val ballGenerator: () => BallRegist with Module = mapping.ballName match {
          case "VecBall"  => () => new VecBall(b)
          case "ReluBall" => () => new ReluBall(b)
          case name       => throw new IllegalArgumentException(s"Unknown ball name: $name")
        }
        ballGenerator
      }
    ) {}
