//===- f32addOps.h - F32AddBall operations --------------------------------===//

#ifndef F32ADD_OPS_H
#define F32ADD_OPS_H

#include "Interfaces/BankInterfaces.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "f32addDialect.h.inc"

#define GET_OP_CLASSES
#include "f32addOps.h.inc"

#endif // F32ADD_OPS_H
