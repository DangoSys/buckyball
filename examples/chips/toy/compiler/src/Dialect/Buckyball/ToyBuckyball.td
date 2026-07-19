//===-- ToyBuckyball.td - Toy Buckyball dialect operations -*- tablegen -*-===//
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
// This is the Toy compiler's Buckyball dialect assembly.
//
//===----------------------------------------------------------------------===//

#ifndef TOY_BUCKYBALL_OPS
#define TOY_BUCKYBALL_OPS

include "Buckyball.td"

include "ToyOps.td"

// Ball-specific operations
// Note: GemminiBall operations are not included yet
include "VecBall.td"
include "ReluBall.td"
include "TransposeBall.td"
include "Im2colBall.td"
include "MatrixBall.td"
include "Fp2IntBall.td"
include "Int2FpBall.td"


#endif // TOY_BUCKYBALL_OPS
