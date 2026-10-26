//===- LowerLinalgToBuckyball.h - Linalg to Buckyball conversion ----------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// The framework collects the marked MegaKernel regions. Each Ball
// claims its own Linalg operations in its convert-linalg-to-<ball>-ball pass,
// and a pipeline chooses which Balls claim; the rest stays in Linalg for the
// CPU.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_CONVERSION_LOWER_LINALG_TO_BUCKYBALL_H
#define BUDDY_CONVERSION_LOWER_LINALG_TO_BUCKYBALL_H

#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace linalg {
class GenericOp;
}
namespace buddy {

void prepareQuantizedWeights(ArrayRef<linalg::GenericOp> stages);
void formMegaKernelRegions(ArrayRef<linalg::GenericOp> stages);

// Collect the regions selected by the design midend.
void populateMegaKernelLinalgLoweringPatterns(RewritePatternSet &patterns);
void registerLowerLinalgToBuckyballPass();

} // namespace buddy
} // namespace mlir

#endif // BUDDY_CONVERSION_LOWER_LINALG_TO_BUCKYBALL_H
