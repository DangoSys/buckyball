//===- IntrinsicsRISCVBuckyball.td - Buckyball intrinsics (main) ---------===//
//
// Main file that includes all Buckyball Ball intrinsics
//
//===----------------------------------------------------------------------===//

// Base intrinsics (common to all Balls)
include "IntrinsicsRISCVBuckyballBase.td"

// Ball-specific intrinsics
include "IntrinsicsRISCVVecBall.td"
include "IntrinsicsRISCVReluBall.td"
include "IntrinsicsRISCVTransposeBall.td"
include "IntrinsicsRISCVIm2colBall.td"
include "IntrinsicsRISCVSystolicArrayBall.td"
include "IntrinsicsRISCVQuantBall.td"
include "IntrinsicsRISCVDequantBall.td"
