//===- mxint8mmOps.h - Mxint8mmBall operations
//------------------------------===//

#ifndef MXINT8MM_OPS_H
#define MXINT8MM_OPS_H

#include "Acc/BankInterfaces.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mxint8mmDialect.h.inc"

#define GET_OP_CLASSES
#include "mxint8mmOps.h.inc"

#endif // MXINT8MM_OPS_H
