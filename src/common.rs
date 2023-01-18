use petgraph::graph::IndexType;
use petgraph::{EdgeType, Graph};

/// Generates an empty graph with `n` nodes and given node and edge capacities.
pub(crate) fn empty_graph_with_capacity<Ty: EdgeType, Ix: IndexType>(
    n: usize,
    node_capacity: usize,
    edge_capacity: usize,
) -> Graph<(), (), Ty, Ix> {
    let mut graph = Graph::with_capacity(node_capacity, edge_capacity);
    for _ in 0..n {
        graph.add_node(());
    }
    graph
}

#[cfg(test)]
pub(crate) fn assert_graph_eq<Ty: EdgeType, Ix: IndexType>(
    graph: &Graph<(), (), Ty, Ix>,
    expected: &Graph<(), (), Ty, Ix>,
) {
    assert_eq!(graph.node_count(), expected.node_count());
    assert_eq!(graph.edge_count(), expected.edge_count());
    for (node_index, expected_index) in graph.node_indices().zip(expected.node_indices()) {
        assert_eq!(graph[node_index], expected[expected_index]);
    }
    for (edge_index, expected_index) in graph.edge_indices().zip(expected.edge_indices()) {
        assert_eq!(
            graph.edge_endpoints(edge_index),
            expected.edge_endpoints(expected_index)
        );
    }
}
