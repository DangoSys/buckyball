#include "Target/BuckyballTargetRegistry.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

#include "BuckyballTargetRegistry.inc"

const buckyball_target::BuckyballTargetConfig &
buckyball_target::getBuckyballTarget() {
  return kBuckyballTarget;
}

static cl::list<std::string> selectedBalls(
    "buckyball-ball", cl::CommaSeparated,
    cl::desc(
        "Select Ball instance names when an instruction has multiple owners"));

static const buckyball_target::BuckyballIsaEntry &
resolveInstruction(StringRef mnemonic) {
  for (const std::string &name : selectedBalls) {
    if (!llvm::any_of(kBuckyballTarget.ballMappings,
                      [&](const auto &ball) { return ball.name == name; }))
      report_fatal_error(Twine("Unknown Ball instance: ") + name);
  }
  const buckyball_target::BuckyballIsaEntry *unique = nullptr;
  const buckyball_target::BuckyballIsaEntry *selected = nullptr;
  unsigned count = 0;
  for (const auto &entry : kBuckyballTarget.isa) {
    if (entry.mnemonic != mnemonic)
      continue;
    unique = &entry;
    ++count;
    auto name = kBuckyballTarget.ballMappings[entry.ballIndex].name;
    if (llvm::is_contained(selectedBalls, name)) {
      if (selected)
        report_fatal_error(
            Twine("Multiple selected Ball instances implement ") + mnemonic);
      selected = &entry;
    }
  }
  if (selected)
    return *selected;
  if (count == 1)
    return *unique;
  if (count > 1)
    report_fatal_error(Twine("Ambiguous instruction ") + mnemonic +
                       "; select its instance with --buckyball-ball=<name>");
  report_fatal_error(Twine("Buckyball core ") + kBuckyballTarget.core +
                     " does not define ISA mnemonic " + mnemonic);
}

int32_t buckyball_target::getBuckyballFunct7(llvm::StringRef mnemonic) {
  return resolveInstruction(mnemonic).funct7;
}

const buckyball_target::BuckyballBallMapping &
buckyball_target::getBuckyballBallMapping(llvm::StringRef mnemonic) {
  return kBuckyballTarget.ballMappings[resolveInstruction(mnemonic).ballIndex];
}

int64_t buckyball_target::getBuckyballBallParam(llvm::StringRef mnemonic,
                                                llvm::StringRef param) {
  for (const BuckyballBallParam &entry :
       getBuckyballBallMapping(mnemonic).params) {
    if (entry.name == param)
      return entry.value;
  }
  report_fatal_error(Twine("Buckyball instruction ") + mnemonic +
                     " has no parameter " + param);
}
