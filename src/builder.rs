use super::network::*;
use super::models::*;

use libfornix::*;

/// A builder pattern that allows building an neural network object
pub struct NeuralNetworkBuilder {
    layer: Layer,
    network: NeuralNetwork,
}

impl NeuralNetworkBuilder {
    /// Creates a new NN builder, and starts the first layer
    /// `inputs` - number of inputs
    /// `outputs` - number of outputs
    pub fn new(inputs: usize, outputs: usize) -> Self {
        NeuralNetworkBuilder {
            layer: Layer::new(),
            network: NeuralNetwork {
                layers: Vec::new(),
            }
        }
    }

    /// Finishes current layer
    pub fn next_layer(mut self) -> Self {
        if self.layer.neurons.len() == 0 {
            warn!("tried to create empty layer");
            return self;
        }

        self.network.layers.push(self.layer);
        self.layer = Layer::new();
        self
    }

    /// Adds a neuron to the current layer
    pub fn add_neuron<T: NeuronModel + 'static>(mut self, neuron: T) -> Self {
        self.layer.neurons.push(Neuron::new(neuron));
        self
    }

    /// Creates a connection
    pub fn connect(from: (usize, usize), to: (usize, usize), weight: f64) {
    }

    /// Creates the network by generating directional connections between layers with random weights
    pub fn create_directional(mut self, rng: &mut RandomNumberProvider) -> NeuralNetwork {
        // end current layer
        if self.layer.neurons.len() > 0 {
            self = self.next_layer();
        }

        // create connections
        for i in 0..self.network.layers.len() {
        }

        self.network
    }
}