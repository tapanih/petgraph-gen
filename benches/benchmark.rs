use criterion::{black_box, criterion_group, criterion_main, Criterion};
use petgraph::prelude::UnGraph;
use petgraph::Graph;
use rand::rngs::SmallRng;
use rand::SeedableRng;

fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = SmallRng::from_entropy();

    c.bench_function("complete_graph", |b| {
        b.iter(|| {
            let graph: Graph<(), ()> = petgraph_gen::complete_graph(black_box(500));
            graph
        })
    });

    c.bench_function("empty_graph", |b| {
        b.iter(|| {
            let graph: Graph<(), ()> = petgraph_gen::empty_graph(black_box(100000));
            graph
        })
    });

    c.bench_function("star_graph", |b| {
        b.iter(|| {
            let graph: Graph<(), ()> = petgraph_gen::star_graph(black_box(100000));
            graph
        })
    });

    c.bench_function("barabasi_albert_graph", |b| {
        b.iter(|| {
            let graph: Graph<(), ()> =
                petgraph_gen::barabasi_albert_graph(&mut rng, black_box(1000), black_box(40), None);
            graph
        })
    });

    c.bench_function("random_gnp_graph", |b| {
        b.iter(|| {
            let graph: Graph<(), ()> =
                petgraph_gen::random_gnp_graph(&mut rng, black_box(250), black_box(0.3));
            graph
        })
    });

    c.bench_function("random_gnm_graph directed sparse", |b| {
        b.iter(|| {
            let graph: Graph<(), ()> =
                petgraph_gen::random_gnm_graph(&mut rng, black_box(250), black_box(3000));
            graph
        })
    });

    c.bench_function("random_gnm_graph directed dense", |b| {
        b.iter(|| {
            let graph: Graph<(), ()> =
                petgraph_gen::random_gnm_graph(&mut rng, black_box(250), black_box(30000));
            graph
        })
    });

    c.bench_function("random_gnm_graph undirected sparse", |b| {
        b.iter(|| {
            let graph: UnGraph<(), ()> =
                petgraph_gen::random_gnm_graph(&mut rng, black_box(250), black_box(3000));
            graph
        })
    });

    c.bench_function("random_gnm_graph undirected dense", |b| {
        b.iter(|| {
            let graph: UnGraph<(), ()> =
                petgraph_gen::random_gnm_graph(&mut rng, black_box(250), black_box(30000));
            graph
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
