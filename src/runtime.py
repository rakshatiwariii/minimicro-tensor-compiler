import numpy as np
from src.ir import Graph

def execute_graph(graph: Graph, feed_dict: dict) -> list:
    """Executes the optimized graph IR using NumPy backend kernels."""
    evaluated = {}

    def _eval(node):
        if node.name in evaluated:
            return evaluated[node.name]

        if node.op_type == 'Constant':
            val = node.attrs['value']
        elif node.op_type == 'Variable':
            val = feed_dict[node.name]
        elif node.op_type == 'Add':
            val = _eval(node.inputs[0]) + _eval(node.inputs[1])
        elif node.op_type == 'Relu':
            val = np.maximum(_eval(node.inputs[0]), 0)
        elif node.op_type == 'FusedAddRelu':
            # Fused execution saves RAM allocation bandwidth
            val = np.maximum(_eval(node.inputs[0]) + _eval(node.inputs[1]), 0)
        else:
            raise NotImplementedError(f"Unsupported op_type: {node.op_type}")

        evaluated[node.name] = val
        return val

    return [_eval(out) for out in graph.outputs]
