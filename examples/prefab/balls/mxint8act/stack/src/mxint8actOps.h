//===- mxint8actOps.h - Mxint8actBall operations ------------------------===//

#ifndef MXINT8ACT_OPS_H
#define MXINT8ACT_OPS_H

#include "Acc/BankInterfaces.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mxint8actDialect.h.inc"

#define GET_OP_CLASSES
#include "mxint8actOps.h.inc"

#endif // MXINT8ACT_OPS_H
