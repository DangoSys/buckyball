//===- Plugin.cpp - The Buckyball compiler as a buddy-opt plugin ----------===//
//
// buddy-opt --load-dialect-plugin=<so> --load-pass-plugin=<so> loads the
// Buckyball dialect and the framework passes. Ball passes register themselves
// when the plugin library is loaded.
//
//===----------------------------------------------------------------------===//

#include "Conversion/LowerBuckyball/LowerBuckyball.h"

#include "mlir/Tools/Plugins/DialectPlugin.h"
#include "mlir/Tools/Plugins/PassPlugin.h"
#include "llvm/Config/llvm-config.h"

#include "Buckyball/BuckyballDialect.h"

using namespace mlir;

extern "C" LLVM_ATTRIBUTE_WEAK DialectPluginLibraryInfo
mlirGetDialectPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "Buckyball", LLVM_VERSION_STRING,
          [](DialectRegistry *registry) {
            registry->insert<::buddy::buckyball::BuckyballDialect>();
          }};
}

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo mlirGetPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "Buckyball", LLVM_VERSION_STRING,
          [] { mlir::buddy::registerBuckyballPasses(); }};
}
