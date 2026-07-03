//===-- BuckyballCarrier.td - Generic Buckyball carrier ops -*- tablegen -*-===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#ifndef BUCKYBALL_CARRIER
#define BUCKYBALL_CARRIER

def InstOp : Buckyball_Op<"inst"> {
  let summary = "Generic Buckyball custom instruction carrier";
  let description = [{
    Emits a Buckyball custom instruction with explicit rs1, rs2 and funct7.
  }];
  let arguments = (ins I64:$rs1, I64:$rs2,
                       ConfinedAttr<I32Attr,
                                    [IntMinValue<0>, IntMaxValue<127>]>:$funct7);
  let assemblyFormat = "$rs1 `,` $rs2 attr-dict `:` type($rs1)";
}

#endif // BUCKYBALL_CARRIER
