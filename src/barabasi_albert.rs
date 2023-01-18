use crate::star_graph;
use petgraph::graph::{IndexType, NodeIndex};
use petgraph::prelude::EdgeRef;
use petgraph::{EdgeType, Graph};
use rand::distributions::Distribution;
use rand::distributions::Uniform;
use rand::Rng;

/// Generates a random graph with `n` nodes using the [Barabási-Albert][ba] model. The process
/// starts with a star graph of `m + 1` nodes or an initial graph given by the `initial_graph`
/// parameter. Then additional nodes are added one by one.
/// Each new node is connected to `m` existing nodes, where the probability of a node
/// being connected to a given node is proportional to the number of edges that node already has.
///
/// # Examples
/// ```
/// use petgraph::Graph;
/// use petgraph_gen::barabasi_albert_graph;
///
/// let mut rng = rand::thread_rng();
/// let graph: Graph<(), ()> = barabasi_albert_graph(&mut rng, 100, 3, None);
/// assert_eq!(graph.node_count(), 100);
/// assert_eq!(graph.edge_count(), 291);
/// ```
///
/// # Panics
/// Panics if `m` equals 0 or is greater than or equal to `n`.
///
/// If an initial graph is provided, panics if the initial graph
/// * has no edges
/// * has less than `m` or more than `n` nodes
///
/// [ba]: https://en.wikipedia.org/wiki/Barab%C3%A1si%E2%80%93Albert_model
pub fn barabasi_albert_graph<
    R: Rng + ?Sized,
    Ty: EdgeType,
    Ix: IndexType,
    G: Into<Option<Graph<(), (), Ty, Ix>>>,
>(
    rng: &mut R,
    n: usize,
    m: usize,
    initial_graph: G,
) -> Graph<(), (), Ty, Ix> {
    assert!(m >= 1, "Parameter m must be greater than 0");
    assert!(m < n, "Parameter m must be less than n");

    let mut graph = initial_graph.into().unwrap_or_else(|| star_graph(m));
    let initial_node_count = graph.node_count();
    assert!(
        graph.edge_count() > 0,
        "Initial graph must have at least one edge"
    );
    assert!(
        initial_node_count >= m,
        "Initial graph must have at least m nodes"
    );
    assert!(
        initial_node_count <= n,
        "Initial graph must have at most n nodes"
    );
    graph.reserve_edges((n - initial_node_count) * m);
    graph.reserve_nodes(n - initial_node_count);

    let mut repeated_nodes = Vec::with_capacity((n - m) * m * 2);
    for node in graph.node_indices() {
        for edge in graph.edges(node) {
            repeated_nodes.push(edge.source());
            repeated_nodes.push(edge.target());
        }
    }

    let mut picked = vec![false; n];
    let mut targets = vec![NodeIndex::new(0); m];

    for _ in initial_node_count..n {
        let node = graph.add_node(());
        let uniform_distribution = Uniform::new(0, repeated_nodes.len());

        let mut i = 0;
        while i < m {
            let random_index = uniform_distribution.sample(rng);
            let target = repeated_nodes[random_index];
            if !picked[target.index()] {
                picked[target.index()] = true;
                targets[i] = target;
                i += 1;
            }
        }
        for target in &targets {
            graph.add_edge(node, *target, ());
            repeated_nodes.push(node);
            repeated_nodes.push(*target);
            picked[target.index()] = false;
        }
    }
    graph
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::complete_graph;
    use petgraph::graph::DiGraph;
    use rand::prelude::SmallRng;
    use rand::SeedableRng;

    #[test]
    fn test_directed_barabasi_albert_graph_nodes_have_at_most_m_outgoing_edges() {
        let mut rng = SmallRng::from_entropy();
        let graph: DiGraph<(), ()> = barabasi_albert_graph(&mut rng, 100, 3, None);
        graph.node_indices().for_each(|node| {
            let outgoing_edges = graph.edges_directed(node, petgraph::Outgoing).count();
            assert!(outgoing_edges <= 3);
        });
    }

    #[test]
    fn test_directed_barabasi_albert_graph_with_initial_graph() {
        let mut rng = SmallRng::from_entropy();
        let graph: DiGraph<(), ()> = barabasi_albert_graph(&mut rng, 100, 3, complete_graph(4));
        assert_eq!(graph.node_count(), 100);
        assert_eq!(graph.edge_count(), 300);
        graph.node_indices().for_each(|node| {
            let outgoing_edges = graph.edges_directed(node, petgraph::Outgoing).count();
            assert_eq!(outgoing_edges, 3);
        });
    }

    #[test]
    fn test_barabasi_albert_graph_with_maximum_sized_initial_graph() {
        let mut rng = SmallRng::from_entropy();
        let star_graph: Graph<(), ()> = barabasi_albert_graph(&mut rng, 5, 4, star_graph(4));
        assert_eq!(star_graph.node_count(), 5);
        assert_eq!(star_graph.edge_count(), 4);
        assert_eq!(star_graph.edges(NodeIndex::new(0)).count(), 4);
    }
}
