package framework.blink

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import freechips.rocketchip.diplomacy._
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import chisel3.experimental.SourceInfo
import framework.builtin.memdomain.mem.{SramReadIO, SramWriteIO}
import examples.toy.balldomain.rs.{BallRsIssue, BallRsComplete}

// Blink protocol parameters
case class BBusParams (sramReadBW: Int = 2, sramWriteBW: Int = 1)  	// DownParam
case class BlinkParams(sramReadBW: Int = 2, sramWriteBW: Int = 1)  	// EdgeParam
case class BallParams (sramReadBW: Int = 2, sramWriteBW: Int = 1)  	// UpParam

// PARAMETER TYPES:                       D              U            E          B
object BlinkNodeImp extends SimpleNodeImp[BBusParams, BallParams, BlinkParams, BlinkBundle] {
  def edge(pd: BBusParams, pu: BallParams, p: Parameters, sourceInfo: SourceInfo) = {
		require(pd.sramReadBW >= pu.sramReadBW, "sramReadBW of BBusParams must be greater than or equal to sramReadBW of BallParams")
		require(pd.sramWriteBW >= pu.sramWriteBW, "sramWriteBW of BBusParams must be greater than or equal to sramWriteBW of BallParams")
    BlinkParams(pd.sramReadBW, pd.sramWriteBW)
  }
  def bundle(e: BlinkParams) = new BlinkBundle(e)(examples.CustomBuckyBallConfig(), Parameters.empty)
  def render(e: BlinkParams) = RenderedEdge("blue", s"width = ${e.sramReadBW}")
}

/** node for [[Ball]] (source) */
class BallNode(widths: Seq[BBusParams])(implicit valName: ValName)
  extends SourceNode(BlinkNodeImp)(widths)

/** node for [[BBus]] (sink) */
class BBusNode(width: BallParams)(implicit valName: ValName)
  extends SinkNode(BlinkNodeImp)(Seq(width))

/** node for [[Blink]] (nexus) */
class BlinkNode(dFn: Seq[BBusParams] => BBusParams,
                uFn: Seq[BallParams] => BallParams)(implicit valName: ValName)
  extends NexusNode(BlinkNodeImp)(dFn, uFn)




class BlinkStatus(implicit p: Parameters) extends Bundle {
	val start  = Bool()
	val arrive = Bool()
	val finish = Bool()
	val iter 	 = UInt(10.W)
}

class BlinkBundle(params: BlinkParams)(implicit b: CustomBuckyBallConfig, p: Parameters) extends Bundle {

	val cmd = new Bundle {
    val req = Flipped(Decoupled(new BallRsIssue))
    val resp = Decoupled(new BallRsComplete)
	}

	val data = new Bundle {
		val sramRead  = Vec(params.sramReadBW, Flipped(new SramReadIO(params.sramReadBW, params.sramWriteBW)))
		val sramWrite = Vec(params.sramWriteBW, Flipped(new SramWriteIO(params.sramReadBW, params.sramWriteBW, params.sramWriteBW)))
	}

	val status = Decoupled(new BlinkStatus())
}


/** blink DUT (nexus) */
class Blink(implicit b: CustomBuckyBallConfig, p: Parameters) extends LazyModule {
  val node = new BlinkNode (
    { case dps: Seq[BBusParams] =>
      require(dps.forall(dp => dp.sramReadBW == dps.head.sramReadBW), "inward, downward adder widths must be equivalent")
      dps.head
    },
    { case ups: Seq[BallParams] =>
      require(ups.forall(up => up.sramReadBW == ups.head.sramReadBW), "outward, upward adder widths must be equivalent")
      ups.head
    }
  )
  
  lazy val module = new LazyModuleImp(this) {
    // 获取协商后的参数
    val edgeParams = node.edges.out.head
    
    // 创建IO接口
    val io = IO(new BlinkBundle(edgeParams))
    
    // Blink主要作为参数协商节点，不进行实际数据处理
    // 这里可以添加状态监控和协商逻辑
    
    // Blink只是协商节点，不进行数据处理
    // 实际的数据传递在更高层次完成
    require(node.in.size >= 1)
  }

  override lazy val desiredName = "Blink"
}