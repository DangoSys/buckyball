//===- mxfp2intOps.h - Mxfp2IntBall operations ----------------------------===//

#ifndef MXFP2INT_OPS_H
#define MXFP2INT_OPS_H

#include "Interfaces/BankInterfaces.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mxfp2intDialect.h.inc"

#define GET_OP_CLASSES
#include "mxfp2intOps.h.inc"

#endif // MXFP2INT_OPS_H
