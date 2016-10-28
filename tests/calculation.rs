extern crate libfornix_neuralnetwork;
extern crate libfornix;

use libfornix_neuralnetwork::*;
use libfornix::*;

#[test]
fn test_execute() {
    let mut rng = OsRandomNumberProvider::new().unwrap();

    // create a tested neural network
    let network = NeuralNetworkBuilder::new(3, 1)
            .add_neuron(TrivialNeuron::create_initial(&mut rng))
            .add_neuron(TrivialNeuron::create_initial(&mut rng))
            .add_neuron(TrivialNeuron::create_initial(&mut rng))
        .next_layer()
            .add_neuron(TrivialNeuron::create_initial(&mut rng))
            .add_neuron(TrivialNeuron::create_initial(&mut rng))
        .create_directional(&mut rng);

    // create inputs
    //let inputs =
}
