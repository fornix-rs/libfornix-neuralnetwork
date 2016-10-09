extern crate libfornix_neuralnetwork;
extern crate libfornix;

use libfornix_neuralnetwork::*;
use libfornix::*;

#[test]
fn test_locate() {
    let mut rng = OsRandomNumberProvider::new().unwrap();

    // create test network
    let mut testTarget = NeuralNetwork {
      layers: Vec::new(),
    };
    testTarget.layers.push(Layer {
        neurons: vec![Neuron::new(TrivialNeuron::create_initial(&mut rng))],
    });

    assert!(testTarget.locate((10, 10)).is_none());
    assert!(testTarget.locate((0, 0)).is_some());
}
