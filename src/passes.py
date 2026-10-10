from src.ir import Node, Graph

def apply_operator_fusion(graph: Graph) -> Graph:
    """Compiler Pass: Merges Add + Relu nodes into FusedAddRelu to eliminate memory round-trips."""
    new_outputs = []
    for out in graph.outputs:
        if out.op_type == 'Relu' and len(out.inputs) == 1:
            producer = out.inputs[0]
            if producer.op_type == 'Add':
                fused_node = Node(
                    name=f"fused_{producer.name}_{out.name}",
                    op_type='FusedAddRelu',
                    inputs=producer.inputs
                )
                new_outputs.append(fused_node)
                continue
        new_outputs.append(out)
    return Graph(new_outputs)
