package examples.designs.gemmini.harts.gemmini.sims

import buckyball.config.Design
import framework.hart.cbus.CBus
import framework.hart.cbus.generated.HartComponents
import framework.hart.hlink.HLinkIO
import framework.device.sims.configs.GlobalConfig
import framework.hart.sims.HartTarget

/** The gemmini design's gemmini hart alone on the framework's single-hart system (sims/hart). */
class HartConfig(design: Design)
    extends HartTarget(GlobalConfig.fromProto(design, CBus.parameters(design, HartComponents.factories))) {
  def instantiateHart(): HLinkIO = CBus.generators(design, b.harts, b.sim, HartComponents.factories).head()
  def hierBlocks = Seq.empty
}
