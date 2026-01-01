package examples.toy.balldomain.bbus

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.instantiable
import framework.top.GlobalConfig
import framework.balldomain.bbus.BBus
import framework.balldomain.blink.BallRegist
import framework.balldomain.prototype.vector.VecBall

/**
 * BBusModule - Ball bus module that directly extends BBus
 */
@instantiable
class BBusModule(b: GlobalConfig)
    extends BBus(
      b,
      b.ballDomain.ballIdMappings.map { mapping =>
        val ballGenerator: () => BallRegist with Module = mapping.ballName match {
          case "VecBall" => () => new VecBall(b, mapping.ballId)
          case name      => throw new IllegalArgumentException(s"Unknown ball name: $name")
        }
        ballGenerator
      }
    ) {}
