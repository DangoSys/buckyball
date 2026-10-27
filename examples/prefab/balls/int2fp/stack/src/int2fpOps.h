//===- int2fpOps.h - Int2FpBall operations --------------------------------===//

#ifndef INT2FP_OPS_H
#define INT2FP_OPS_H

#include "Interfaces/BankInterfaces.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "int2fpDialect.h.inc"

#define GET_OP_CLASSES
#include "int2fpOps.h.inc"

#endif // INT2FP_OPS_H
