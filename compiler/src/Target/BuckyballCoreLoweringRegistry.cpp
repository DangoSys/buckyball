//===- BuckyballCoreLoweringRegistry.cpp - Core lowering providers --------===//

#include "Target/BuckyballCoreLoweringRegistry.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"

namespace {
llvm::StringMap<const mlir::buddy::BuckyballCoreLoweringProvider *> &
providers() {
  static llvm::StringMap<const mlir::buddy::BuckyballCoreLoweringProvider *>
      registry;
  return registry;
}
} // namespace

void mlir::buddy::registerBuckyballCoreLoweringProvider(
    const BuckyballCoreLoweringProvider &provider) {
  if (provider.core.empty())
    llvm::report_fatal_error("Buckyball Core lowering provider has no name");
  auto [it, inserted] = providers().try_emplace(provider.core, &provider);
  if (!inserted)
    llvm::report_fatal_error(llvm::Twine("duplicate Buckyball Core lowering "
                                         "provider: ") +
                             provider.core);
}

const mlir::buddy::BuckyballCoreLoweringProvider &
mlir::buddy::getBuckyballCoreLoweringProvider(llvm::StringRef core) {
  auto it = providers().find(core);
  if (it == providers().end())
    llvm::report_fatal_error(llvm::Twine("no Buckyball lowering provider for "
                                         "Core: ") +
                             core);
  return *it->second;
}
