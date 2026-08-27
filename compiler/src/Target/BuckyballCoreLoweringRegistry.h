//===- BuckyballCoreLoweringRegistry.h - Core lowering providers -*- C++
//-*-===//

#ifndef BUCKYBALL_CORE_LOWERING_REGISTRY_H
#define BUCKYBALL_CORE_LOWERING_REGISTRY_H

#include "Conversion/LowerBuckyball/LowerBuckyball.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::buddy {

struct BuckyballCoreLoweringProvider {
  using TileLoweringFn = void (*)(ConversionTarget &, RewritePatternSet &,
                                  int64_t, int64_t, int64_t);
  using BankSSALoweringFn = void (*)(RewritePatternSet &);
  using PhysicalBankLoweringFn = void (*)(RewritePatternSet &,
                                          PhysicalBankState &);
  using LLVMExportLoweringFn = void (*)(LLVMTypeConverter &,
                                        RewritePatternSet &,
                                        LLVMConversionTarget &, int64_t,
                                        int64_t, int64_t, bool, bool, bool);

  llvm::StringRef core;
  TileLoweringFn populateTileLowering = nullptr;
  BankSSALoweringFn populateBankSSALowering = nullptr;
  PhysicalBankLoweringFn populatePhysicalBankLowering = nullptr;
  LLVMExportLoweringFn populateLLVMExportLowering = nullptr;
};

void registerBuckyballCoreLoweringProvider(
    const BuckyballCoreLoweringProvider &provider);
const BuckyballCoreLoweringProvider &
getBuckyballCoreLoweringProvider(llvm::StringRef core);

} // namespace mlir::buddy

#endif // BUCKYBALL_CORE_LOWERING_REGISTRY_H
