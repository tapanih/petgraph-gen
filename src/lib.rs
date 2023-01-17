//! `petgraph-gen` is a crate that extends [petgraph](https://github.com/petgraph/petgraph)
//! with functions that generate graphs with different properties.

mod barabasi_albert;
mod classic;
mod common;
mod erdos_renyi;

pub use self::barabasi_albert::barabasi_albert_graph;
pub use self::classic::complete_graph;
pub use self::classic::empty_graph;
pub use self::classic::star_graph;
pub use self::erdos_renyi::random_gnm_graph;
pub use self::erdos_renyi::random_gnp_graph;
