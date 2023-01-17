use crate::common::empty_graph_with_capacity;
use petgraph::graph::{IndexType, NodeIndex};
use petgraph::{EdgeType, Graph};

/// Generates a complete graph with `n` nodes. A complete graph is a graph where
/// each node is connected to every other node. On a directed graph, this means
/// that each node has `n - 1` incoming edges and `n - 1` outgoing edges.
///
/// # Examples
/// ```
/// use petgraph_gen::complete_graph;
/// use petgraph::{Directed, Graph, Undirected};
///
/// let directed_graph: Graph<(), (), Directed> = complete_graph(10);
/// assert_eq!(directed_graph.node_count(), 10);
/// assert_eq!(directed_graph.edge_count(), 90);
///
/// let undirected_graph: Graph<(), (), Undirected> = complete_graph(10);
/// assert_eq!(undirected_graph.node_count(), 10);
/// assert_eq!(undirected_graph.edge_count(), 45);
/// ```
///
pub fn complete_graph<Ty: EdgeType, Ix: IndexType>(n: usize) -> Graph<(), (), Ty, Ix> {
    let mut graph = empty_graph_with_capacity(n, n * n);
    if n <= 1 {
        return graph;
    }
    for i in 0..n - 1 {
        for j in i + 1..n {
            graph.add_edge(NodeIndex::new(i), NodeIndex::new(j), ());
            if Ty::is_directed() {
                graph.add_edge(NodeIndex::new(j), NodeIndex::new(i), ());
            }
        }
    }
    graph
}

/// Generates an empty graph with `n` nodes and no edges.
///
/// # Examples
/// ```
/// use petgraph::Graph;
/// use petgraph_gen::empty_graph;
///
/// let graph: Graph<(), ()> = empty_graph(5);
/// assert_eq!(graph.node_count(), 5);
/// assert_eq!(graph.edge_count(), 0);
/// ```
pub fn empty_graph<Ty: EdgeType, Ix: IndexType>(n: usize) -> Graph<(), (), Ty, Ix> {
    empty_graph_with_capacity(n, 0)
}

/// Generates a star graph with a single center node connected to `n` other nodes. The resulting
/// graph has `n + 1` nodes and `n` edges.
///
/// # Examples
/// ```
/// use petgraph_gen::star_graph;
/// use petgraph::{Directed, Graph, Undirected};
/// use petgraph::graph::NodeIndex;
/// use petgraph::visit::EdgeRef;
///
/// let graph: Graph<(), ()> = star_graph(10);
/// assert_eq!(graph.node_count(), 11);
/// assert_eq!(graph.edge_count(), 10);
/// let center_node = NodeIndex::new(0);
/// assert_eq!(graph.edges(center_node).count(), 10);
/// for edge in graph.edges(center_node) {
///    assert_eq!(edge.source(), center_node);
/// }
///
pub fn star_graph<Ty: EdgeType, Ix: IndexType>(n: usize) -> Graph<(), (), Ty, Ix> {
    let mut graph = Graph::with_capacity(n + 1, n);
    let center = graph.add_node(());
    for _ in 0..n {
        let node = graph.add_node(());
        graph.add_edge(center, node, ());
    }
    graph
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::assert_graph_eq;
    use petgraph::graph::{DiGraph, UnGraph};

    #[test]
    fn test_complete_directed_graph() {
        let graph: DiGraph<(), (), u32> = complete_graph(4);
        let expected = Graph::from_edges(&[
            (0, 1),
            (1, 0),
            (0, 2),
            (2, 0),
            (0, 3),
            (3, 0),
            (1, 2),
            (2, 1),
            (1, 3),
            (3, 1),
            (2, 3),
            (3, 2),
        ]);
        assert_graph_eq(&graph, &expected);
    }

    #[test]
    fn test_complete_undirected_graph() {
        let graph: UnGraph<(), (), u16> = complete_graph(4);
        let expected = Graph::from_edges(&[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]);
        assert_graph_eq(&graph, &expected);
    }
}
