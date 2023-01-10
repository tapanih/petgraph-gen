use criterion::{black_box, criterion_group, criterion_main, Criterion};
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
                petgraph_gen::barabasi_albert_graph(&mut rng, black_box(500), black_box(20));
            graph
        })
    });

    c.bench_function("erdos_renyi_graph", |b| {
        b.iter(|| {
            let graph: Graph<(), ()> =
                petgraph_gen::erdos_renyi_graph(&mut rng, black_box(250), black_box(0.3));
            graph
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
