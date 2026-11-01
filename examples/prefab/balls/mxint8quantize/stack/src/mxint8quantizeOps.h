//===- mxint8quantizeOps.h - Mxint8quantizeBall operations ------------===//

#ifndef MXINT8QUANTIZE_OPS_H
#define MXINT8QUANTIZE_OPS_H

#include "Acc/BankInterfaces.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mxint8quantizeDialect.h.inc"

#define GET_OP_CLASSES
#include "mxint8quantizeOps.h.inc"

#endif // MXINT8QUANTIZE_OPS_H
