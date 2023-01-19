use crate::common::empty_graph_with_capacity;
use crate::complete_graph;
use petgraph::graph::{IndexType, NodeIndex};
use petgraph::{EdgeType, Graph};
use rand::distributions::Distribution;
use rand::distributions::{Bernoulli, Uniform};
use rand::Rng;
use rustc_hash::FxHashSet;
use std::mem::swap;

fn dense_random_gnm_graph<R: Rng + ?Sized, Ty: EdgeType, Ix: IndexType>(
    rng: &mut R,
    n: usize,
    m: usize,
) -> Graph<(), (), Ty, Ix> {
    let mut graph = empty_graph_with_capacity(n, n, m);

    let uniform_distribution = Uniform::new(0, m);
    let mut edges = Vec::with_capacity(m);

    let mut edge_count = 0;
    for i in 0..n - 1 {
        for j in i + 1..n {
            edge_count += 1;
            if edge_count <= m {
                edges.push((i, j));
            } else if rng.gen_range(0..edge_count) < m {
                edges[uniform_distribution.sample(rng)] = (i, j);
            }
            if Ty::is_directed() {
                edge_count += 1;
                if edge_count <= m {
                    edges.push((j, i));
                } else if rng.gen_range(0..edge_count) < m {
                    edges[uniform_distribution.sample(rng)] = (j, i);
                }
            }
        }
    }
    for (source, target) in edges {
        graph.add_edge(NodeIndex::new(source), NodeIndex::new(target), ());
    }
    graph
}

fn sparse_random_gnm_graph<R: Rng + ?Sized, Ty: EdgeType, Ix: IndexType>(
    rng: &mut R,
    n: usize,
    m: usize,
) -> Graph<(), (), Ty, Ix> {
    let mut graph = empty_graph_with_capacity(n, n, m);

    let uniform_distribution = Uniform::new(0, n);
    let mut edges = FxHashSet::default();
    while edges.len() < m {
        let mut source = uniform_distribution.sample(rng);
        let mut target = uniform_distribution.sample(rng);
        if source == target {
            continue;
        }
        if !Ty::is_directed() && source > target {
            swap(&mut source, &mut target);
        }
        edges.insert((source, target));
    }
    for (source, target) in edges {
        graph.add_edge(NodeIndex::new(source), NodeIndex::new(target), ());
    }
    graph
}

/// Generates a random graph according to the `G(n,m)` Erdős-Rényi model. The resulting graph has `n`
/// nodes and `m` edges are selected randomly and uniformly from the set of all possible edges
/// (excluding loop edges).
///
/// # Examples
/// ```
/// use petgraph_gen::random_gnm_graph;
/// use petgraph::graph::{DiGraph, UnGraph};
/// use rand::SeedableRng;
/// use rand::rngs::SmallRng;
///
/// let mut rng = SmallRng::from_entropy();
/// let undirected_graph: UnGraph<(), ()> = random_gnm_graph(&mut rng, 10, 20);
/// assert_eq!(undirected_graph.node_count(), 10);
/// assert_eq!(undirected_graph.edge_count(), 20);
///
/// let directed_graph: DiGraph<(), ()> = random_gnm_graph(&mut rng, 10, 50);
/// assert_eq!(directed_graph.node_count(), 10);
/// assert_eq!(directed_graph.edge_count(), 50);
/// ```
///
/// # Panics
/// Panics if `m` is greater than the number of possible edges.
pub fn random_gnm_graph<R: Rng + ?Sized, Ty: EdgeType, Ix: IndexType>(
    rng: &mut R,
    n: usize,
    m: usize,
) -> Graph<(), (), Ty, Ix> {
    let max_edges = if Ty::is_directed() {
        n * (n - 1)
    } else {
        n * (n - 1) / 2
    };
    assert!(
        m <= max_edges,
        "Parameter m must be less than or equal to {}",
        max_edges
    );

    if m == max_edges {
        complete_graph(n)
    } else if m < max_edges / 6 {
        // based on some experiments, the sparse algorithm is faster when less than 15 % of the edges are selected
        sparse_random_gnm_graph(rng, n, m)
    } else {
        dense_random_gnm_graph(rng, n, m)
    }
}

/// Generates a random graph according to the `G(n,p)` Erdős-Rényi model.
/// The resulting graph has `n` nodes and edges are selected with probability `p` from the set
/// of all possible edges (excluding loop edges).
///
/// # Examples
/// ```
/// use petgraph_gen::random_gnp_graph;
/// use petgraph::graph::{DiGraph, UnGraph};
/// use rand::SeedableRng;
///
/// let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
/// let undirected_graph: UnGraph<(), ()> = random_gnp_graph(&mut rng, 10, 0.3);
/// assert_eq!(undirected_graph.node_count(), 10);
/// assert_eq!(undirected_graph.edge_count(), 15); // out of 45 possible edges
///
/// let directed_graph: DiGraph<(), ()> = random_gnp_graph(&mut rng, 10, 0.5);
/// assert_eq!(directed_graph.node_count(), 10);
/// assert_eq!(directed_graph.edge_count(), 40); // out of 90 possible edges
/// ```
pub fn random_gnp_graph<R: Rng + ?Sized, Ty: EdgeType, Ix: IndexType>(
    rng: &mut R,
    n: usize,
    p: f64,
) -> Graph<(), (), Ty, Ix> {
    let mut graph = Graph::with_capacity(n, ((n * n) as f64 * p) as usize);
    let nodes: Vec<_> = (0..n).map(|_| graph.add_node(())).collect();
    if p <= 0.0 {
        return graph;
    }
    if p >= 1.0 {
        return complete_graph(n);
    }
    let bernoulli_distribution = Bernoulli::new(p).unwrap();

    for (i, node) in nodes.iter().enumerate() {
        for other_node in nodes.iter().skip(i + 1) {
            if bernoulli_distribution.sample(rng) {
                graph.add_edge(*node, *other_node, ());
            }
            if Ty::is_directed() && bernoulli_distribution.sample(rng) {
                graph.add_edge(*other_node, *node, ());
            }
        }
    }
    graph
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::assert_graph_eq;
    use petgraph::graph::{DiGraph, UnGraph};
    use rand::rngs::mock::StepRng;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;
    use std::collections::HashSet;

    #[test]
    fn test_directed_random_gnm_graph_does_not_have_self_loops() {
        let mut rng = SmallRng::from_entropy();
        let graph: DiGraph<(), ()> = random_gnm_graph(&mut rng, 10, 89);
        for edge in graph.raw_edges() {
            assert_ne!(edge.source(), edge.target()); // no self-loops
        }
    }

    #[test]
    fn test_undirected_random_gnm_graph_does_not_have_self_loops() {
        let mut rng = SmallRng::from_entropy();
        let graph: UnGraph<(), ()> = random_gnm_graph(&mut rng, 10, 44);
        for edge in graph.raw_edges() {
            assert_ne!(edge.source(), edge.target()); // no self-loops
        }
    }

    #[test]
    fn test_directed_random_gnm_graph_does_not_have_duplicate_edges() {
        let mut rng = SmallRng::from_entropy();
        let graph: DiGraph<(), ()> = random_gnm_graph(&mut rng, 10, 89);
        let mut unique_edges = HashSet::new();
        for edge in graph.raw_edges() {
            let source = edge.source().index();
            let target = edge.target().index();
            assert!(unique_edges.insert((source, target)));
        }
    }

    #[test]
    fn test_undirected_random_gnm_graph_does_not_have_duplicate_edges() {
        let mut rng = SmallRng::from_entropy();
        let graph: UnGraph<(), ()> = random_gnm_graph(&mut rng, 10, 44);
        let mut unique_edges = HashSet::new();
        for edge in graph.raw_edges() {
            let source = edge.source().index();
            let target = edge.target().index();
            assert!(unique_edges.insert((source, target)));
            assert!(unique_edges.insert((target, source)));
        }
    }

    #[test]
    fn test_undirected_random_gnm_graph_with_zero_edges() {
        let mut rng = SmallRng::from_entropy();
        let graph: UnGraph<(), ()> = random_gnm_graph(&mut rng, 100, 0);
        assert_eq!(graph.node_count(), 100);
        assert_eq!(graph.edge_count(), 0);
    }

    #[test]
    fn test_directed_random_gnm_graph_with_zero_edges() {
        let mut rng = SmallRng::from_entropy();
        let graph: DiGraph<(), ()> = random_gnm_graph(&mut rng, 100, 0);
        assert_eq!(graph.node_count(), 100);
        assert_eq!(graph.edge_count(), 0);
    }

    #[test]
    fn test_undirected_random_gnm_graph_with_maximum_edges() {
        let mut rng = SmallRng::from_entropy();
        let graph: UnGraph<(), ()> = random_gnm_graph(&mut rng, 10, 45);
        assert_eq!(graph.node_count(), 10);
        assert_eq!(graph.edge_count(), 45);
    }

    #[test]
    fn test_directed_random_gnm_graph_with_maximum_edges() {
        let mut rng = SmallRng::from_entropy();
        let graph: DiGraph<(), ()> = random_gnm_graph(&mut rng, 10, 90);
        assert_eq!(graph.node_count(), 10);
        assert_eq!(graph.edge_count(), 90);
    }

    #[test]
    fn test_empty_undirected_random_gnp_graph() {
        let mut rng = rand::thread_rng();
        let graph: UnGraph<(), ()> = random_gnp_graph(&mut rng, 10, 0.0);
        assert_eq!(graph.node_count(), 10);
        assert_eq!(graph.edge_count(), 0);
    }

    #[test]
    fn test_complete_undirected_random_gnp_graph() {
        let mut rng = rand::thread_rng();
        let graph: UnGraph<(), ()> = random_gnp_graph(&mut rng, 10, 1.0);
        assert_eq!(graph.node_count(), 10);
        assert_eq!(graph.edge_count(), 45);
    }

    #[test]
    fn test_complete_directed_random_gnp_graph() {
        let mut rng = rand::thread_rng();
        let graph: DiGraph<(), ()> = random_gnp_graph(&mut rng, 10, 1.0);
        assert_eq!(graph.node_count(), 10);
        assert_eq!(graph.edge_count(), 90);
    }

    #[test]
    fn test_random_gnp_graph() {
        let mut rng = StepRng::new(0, u64::MAX / 2 + 1);
        let graph: UnGraph<(), ()> = random_gnp_graph(&mut rng, 5, 0.5);
        let expected = Graph::from_edges(&[(0, 1), (0, 3), (1, 2), (1, 4), (2, 4)]);
        assert_graph_eq(&graph, &expected);
    }

    #[test]
    #[should_panic(expected = "m must be less than or equal to 90")]
    fn test_random_gnm_graph_panic() {
        let mut rng = rand::thread_rng();
        let _: Graph<(), ()> = random_gnm_graph(&mut rng, 10, 91);
    }
}
