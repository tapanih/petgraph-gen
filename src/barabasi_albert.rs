use petgraph::graph::IndexType;
use petgraph::{EdgeType, Graph};
use rand::distributions::Distribution;
use rand::distributions::Uniform;
use rand::Rng;
use std::collections::HashSet;

/// Generates a random graph with `n` nodes using the [Barabási-Albert][ba] model. The graph starts
/// with an empty graph of `m` nodes and adds `n - m` additional nodes. Each new node is connected to `m`
/// existing nodes, where the probability of a node being connected to a given node is proportional
/// to the number of edges that node already has.
///
/// # Examples
/// ```
/// use petgraph::Graph;
/// use petgraph_gen::barabasi_albert_graph;
///
/// let mut rng = rand::thread_rng();
/// let graph: Graph<(), ()> = barabasi_albert_graph(&mut rng, 100, 3);
/// assert_eq!(graph.node_count(), 100);
/// assert_eq!(graph.edge_count(), 291);
/// ```
///
/// # Panics
/// Panics if `m` equals 0 or is greater than or equal to `n`.
///
/// [ba]: https://en.wikipedia.org/wiki/Barab%C3%A1si%E2%80%93Albert_model
pub fn barabasi_albert_graph<R: Rng + ?Sized, Ty: EdgeType, Ix: IndexType>(
    rng: &mut R,
    n: usize,
    m: usize,
) -> Graph<(), (), Ty, Ix> {
    assert!(m >= 1);
    assert!(m < n);

    let mut graph = Graph::with_capacity(n, (n - m) * m);
    let mut repeated_nodes = Vec::with_capacity((n - m) * m);
    for _ in 0..m {
        let node = graph.add_node(());
        repeated_nodes.push(node);
    }

    for _ in m..n {
        let node = graph.add_node(());
        let uniform_distribution = Uniform::new(0, repeated_nodes.len());

        let mut targets = HashSet::new();
        while targets.len() < m {
            let random_index = uniform_distribution.sample(rng);
            targets.insert(repeated_nodes[random_index]);
        }
        for target in targets {
            graph.add_edge(node, target, ());
            repeated_nodes.push(node);
            repeated_nodes.push(target);
        }
    }
    graph
}

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::graph::DiGraph;
    use rand::prelude::SmallRng;
    use rand::SeedableRng;

    #[test]
    fn test_directed_barabasi_albert_graph_has_at_most_m_outgoing_edges() {
        let mut rng = SmallRng::from_entropy();
        let graph: DiGraph<(), ()> = barabasi_albert_graph(&mut rng, 100, 3);
        graph.node_indices().for_each(|node| {
            let outgoing_edges = graph.edges_directed(node, petgraph::Outgoing).count();
            assert!(outgoing_edges <= 3);
        });
    }
}
