//===- mxmmOps.h - MxmmBall operations ------------------------------------===//

#ifndef MXMM_OPS_H
#define MXMM_OPS_H

#include "Interfaces/BankInterfaces.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mxmmDialect.h.inc"

#define GET_OP_CLASSES
#include "mxmmOps.h.inc"

#endif // MXMM_OPS_H
