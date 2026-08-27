#include "Target/BuckyballTargetRegistry.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

namespace {

cl::opt<std::string>
    buckyballTarget("target", cl::desc("Required Buckyball compiler target"),
                    cl::value_desc("target"));

#include "BuckyballTargetRegistry.inc"

} // namespace

const buckyball_target::BuckyballTargetConfig &
buckyball_target::getBuckyballTarget() {
  if (buckyballTarget.empty())
    report_fatal_error("--target is required for Buckyball lowering");
  for (const BuckyballTargetConfig &target : kBuckyballTargets) {
    if (target.name == buckyballTarget)
      return target;
  }
  report_fatal_error(Twine("unknown Buckyball target: ") + buckyballTarget);
}

int32_t buckyball_target::getBuckyballFunct7(llvm::StringRef mnemonic) {
  const BuckyballTargetConfig &target = getBuckyballTarget();
  for (const BuckyballIsaEntry &entry : target.isa) {
    if (entry.mnemonic == mnemonic)
      return entry.funct7;
  }
  report_fatal_error(Twine("Buckyball target ") + target.name +
                     " does not define ISA mnemonic " + mnemonic);
}

void buckyball_target::requireBuckyballBall(llvm::StringRef ballName) {
  const BuckyballTargetConfig &target = getBuckyballTarget();
  for (llvm::StringRef enabled : target.balls) {
    if (enabled == ballName)
      return;
  }
  report_fatal_error(Twine("Buckyball target ") + target.name +
                     " does not enable Ball " + ballName);
}

const buckyball_target::BuckyballBallMapping &
buckyball_target::getBuckyballBallMapping(llvm::StringRef ballName) {
  const BuckyballTargetConfig &target = getBuckyballTarget();
  for (const BuckyballBallMapping &mapping : target.ballMappings) {
    if (mapping.name == ballName)
      return mapping;
  }
  report_fatal_error(Twine("Buckyball target ") + target.name +
                     " does not map Ball " + ballName);
}
