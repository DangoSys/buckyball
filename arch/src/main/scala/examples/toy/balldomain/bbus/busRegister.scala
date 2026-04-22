package examples.toy.balldomain.bbus

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.instantiable
import framework.top.GlobalConfig
import framework.balldomain.bbus.BBus
import framework.balldomain.blink.HasBlink
import framework.balldomain.prototype.vector.VecBall
import framework.balldomain.prototype.relu.ReluBall
import framework.balldomain.prototype.transpose.TransposeBall
import framework.balldomain.prototype.im2col.Im2colBall
import framework.balldomain.prototype.systolicarray.SystolicArrayBall
import framework.balldomain.prototype.quant.QuantBall
import framework.balldomain.prototype.dequant.DequantBall
import framework.balldomain.prototype.gemmini.GemminiBall
import framework.balldomain.prototype.trace.TraceBall
import framework.balldomain.prototype.mxfp.MxfpBall
/**
 * BBusModule - Ball bus module that directly extends BBus
 */
@instantiable
class BBusModule(b: GlobalConfig)
    extends BBus(
      b,
      b.ballDomain.ballIdMappings.map { mapping =>
        val ballGenerator: () => HasBlink with Module = mapping.ballName match {
          case "VecBall"           => () => new VecBall(b)
          case "ReluBall"          => () => new ReluBall(b)
          case "TransposeBall"     => () => new TransposeBall(b)
          case "Im2colBall"        => () => new Im2colBall(b)
          case "SystolicArrayBall" => () => new SystolicArrayBall(b)
          case "QuantBall"         => () => new QuantBall(b)
          case "DequantBall"       => () => new DequantBall(b)
          case "GemminiBall"       => () => new GemminiBall(b)
          case "TraceBall"         => () => new TraceBall(b)
          case "MxfpBall"          => () => new MxfpBall(b)
          case name                => throw new IllegalArgumentException(s"Unknown ball name: $name")
        }
        ballGenerator
      }
    ) {}
