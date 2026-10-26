from pathlib import Path
from buddy.compiler.trace import TraceConfig, load_trace_config
from buddy.compiler.graph.transform.trace import TraceInsertionPass
from .operation import QuantizedOp


def trace(graph, path):
    TraceInsertionPass(TraceConfig(load_trace_config(Path(path))))(graph)
    for node in graph.body:
        if isinstance(node, QuantizedOp):
            node.quantization_trace_meta = node.trace_meta
            node.trace_meta = None
