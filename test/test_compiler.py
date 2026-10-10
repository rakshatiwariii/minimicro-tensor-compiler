import numpy as np
from src.ir import Node, Graph
from src.passes import apply_operator_fusion
from src.runtime import execute_graph

def test_operator_fusion_correctness():
    x = Node('x', 'Variable')
    c = Node('c', 'Constant', attrs={'value': np.array([1.0, -2.0, 3.0])})
    add = Node('add', 'Add', inputs=[x, c])
    relu = Node('relu', 'Relu', inputs=[add])
    
    graph = Graph(outputs=[relu])
    optimized = apply_operator_fusion(graph)
    
    assert optimized.outputs[0].op_type == 'FusedAddRelu'
    
    feed = {'x': np.array([0.0, 1.0, -4.0])}
    res = execute_graph(optimized, feed)
    
    expected = np.maximum(feed['x'] + c.attrs['value'], 0)
    np.testing.assert_array_equal(res[0], expected)
