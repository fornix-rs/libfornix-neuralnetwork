extern crate libfornix_neuralnetwork;
extern crate libfornix;

use libfornix_neuralnetwork::*;
use libfornix::*;

#[test]
fn test_trivial() {
    let mut rng = OsRandomNumberProvider::new().unwrap();
    let mut model = TrivialNeuron::create_initial(&mut rng);
    let bias = model.bias;

    {
        let tweakable = model.tweakable_values();

        assert_eq!(1, tweakable.len());
        assert_eq!(bias, *tweakable[0].0);
    }

    let inputs = vec![1.0, 2.0, 3.0];
    assert_eq!(6.0 + bias, model.calculate(&inputs));
}

#[test]
fn test_input() {
    let mut neuron = InputNeuron {
        value: 0.0,
    };

    assert_eq!(0.0, neuron.calculate(&vec![1.0, 2.0, 3.0]));
    assert_eq!(0, neuron.tweakable_values().len());
}

#[test]
fn test_output() {
    let mut neuron = OutputNeuron {};

    assert_eq!(6.0, neuron.calculate(&vec![1.0, 2.0, 3.0]));
    assert_eq!(0, neuron.tweakable_values().len());
}
