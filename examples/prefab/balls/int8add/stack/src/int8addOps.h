//===- int8addOps.h - Int8AddBall operations ------------------------------===//

#ifndef INT8ADD_OPS_H
#define INT8ADD_OPS_H

#include "Interfaces/BankInterfaces.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "int8addDialect.h.inc"

#define GET_OP_CLASSES
#include "int8addOps.h.inc"

#endif // INT8ADD_OPS_H
