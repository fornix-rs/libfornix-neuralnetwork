extern crate libfornix_neuralnetwork;
extern crate libfornix;

use libfornix_neuralnetwork::*;
use libfornix::*;

#[test]
fn test_directional() {
    let mut rng = OsRandomNumberProvider::new().unwrap();

    // Should create a neural network with 2 layers [3, 2]
    let network = NeuralNetworkBuilder::new(3, 1)
            .add_neuron(TrivialNeuron::create_initial(&mut rng))
            .add_neuron(TrivialNeuron::create_initial(&mut rng))
            .add_neuron(TrivialNeuron::create_initial(&mut rng))
        .next_layer()
            .add_neuron(TrivialNeuron::create_initial(&mut rng))
            .add_neuron(TrivialNeuron::create_initial(&mut rng))
        .create_directional(&mut rng);

    // test layout

    // should have 2 layers
    assert_eq!(4, network.layers.len());
    // first layer should have 3 neurons
    assert_eq!(3, network.layers[1].neurons.len());
    // second layer should have 2 neurons
    assert_eq!(2, network.layers[2].neurons.len());

    // test connections input -> layer 1
    assert_eq!(3, network.layers[0].neurons[0].connections.len());
    assert_eq!((1, 0), network.layers[0].neurons[0].connections[0].link);
    assert_eq!((1, 1), network.layers[0].neurons[0].connections[1].link);
    assert_eq!((1, 2), network.layers[0].neurons[0].connections[2].link);
    assert_eq!(3, network.layers[0].neurons[1].connections.len());
    assert_eq!((1, 0), network.layers[0].neurons[1].connections[0].link);
    assert_eq!((1, 1), network.layers[0].neurons[1].connections[1].link);
    assert_eq!((1, 2), network.layers[0].neurons[1].connections[2].link);
    assert_eq!(3, network.layers[0].neurons[2].connections.len());
    assert_eq!((1, 0), network.layers[0].neurons[2].connections[0].link);
    assert_eq!((1, 1), network.layers[0].neurons[2].connections[1].link);
    assert_eq!((1, 2), network.layers[0].neurons[2].connections[2].link);

    // test connections layer 1 -> layer 2
    assert_eq!(2, network.layers[1].neurons[0].connections.len());
    assert_eq!((2, 0), network.layers[1].neurons[0].connections[0].link);
    assert_eq!((2, 1), network.layers[1].neurons[0].connections[1].link);
    assert_eq!(2, network.layers[1].neurons[1].connections.len());
    assert_eq!((2, 0), network.layers[1].neurons[1].connections[0].link);
    assert_eq!((2, 1), network.layers[1].neurons[1].connections[1].link);
    assert_eq!(2, network.layers[1].neurons[2].connections.len());
    assert_eq!((2, 0), network.layers[1].neurons[2].connections[0].link);
    assert_eq!((2, 1), network.layers[1].neurons[2].connections[1].link);

    // test connections layer 2 -> output
    assert_eq!(1, network.layers[2].neurons[0].connections.len());
    assert_eq!((3, 0), network.layers[2].neurons[0].connections[0].link);
    assert_eq!(1, network.layers[2].neurons[1].connections.len());
    assert_eq!((3, 0), network.layers[2].neurons[1].connections[0].link);
}
