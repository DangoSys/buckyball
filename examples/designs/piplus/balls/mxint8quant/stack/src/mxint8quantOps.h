//===- mxint8quantOps.h - Mxint8quantBall operations
//------------------------------===//

#ifndef MXINT8QUANT_OPS_H
#define MXINT8QUANT_OPS_H

#include "Acc/BankInterfaces.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mxint8quantDialect.h.inc"

#define GET_OP_CLASSES
#include "mxint8quantOps.h.inc"

#endif // MXINT8QUANT_OPS_H
