pub mod lib;
use lib::{activations::SIGMOID, network::Network};

// 0, 0 -> 0
// 0, 1 -> 1
// 1, 0 -> 1
// 0, 0 -> 0

fn main() {
    let inputs = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
    ];
    let target = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

    let mut network = Network::new([2, 3, 1].to_vec(), 0.8, SIGMOID);

    network.train(inputs, target, 10000);

    println!("0 and 0: {:?}", network.feed_forward(vec![0.0, 0.0]));
    println!("0 and 1: {:?}", network.feed_forward(vec![0.0, 1.0]));
    println!("1 and 0: {:?}", network.feed_forward(vec![1.0, 0.0]));
    println!("1 and 1: {:?}", network.feed_forward(vec![1.0, 1.0]));
}
