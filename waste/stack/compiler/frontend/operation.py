# ===- operation.py ------------------------------------------------------------
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===---------------------------------------------------------------------------
#
# This is the operation structure of Buddy Compiler graph representation.
#
# ===---------------------------------------------------------------------------

from buddy.compiler.graph.operation import Conv2dOp, Op, OpType


class QuantizedOp(Op):
    def __init__(self):
        super().__init__()
        self.quantization_trace_meta = None


class QuantizedMatmulOp(QuantizedOp):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReduceType


class QuantizedConv2dOp(QuantizedOp):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReduceType
        self._output_nhwc = False


class QuantizedDepthwiseConv2dOp(QuantizedOp):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReduceType
        self._output_nhwc = False


class QuantizedMaxPool2dOp(QuantizedOp):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReduceType


class QuantizedGlobalAvgPoolOp(QuantizedOp):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReduceType


class QuantizedChannelSliceOp(QuantizedOp):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReshapeType


class QuantizedChannelConcatOp(QuantizedOp):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ConcatType


class QuantizedResizeNearestOp(QuantizedOp):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReshapeType


class QuantizedMulOp(QuantizedOp):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class QuantizedAddOp(QuantizedOp):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class Int8QuantizeOp(Op):
    def __init__(self):
        super().__init__()
        self._op_type = OpType.ElementwiseType


class Int8Conv2dOp(Conv2dOp):
    """Signed INT8 convolution with INT32 accumulation and FP32 rescaling."""


class Int8MatmulOp(Op):
    """Signed INT8 matrix product, scales and optional FP32 bias."""

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReduceType
