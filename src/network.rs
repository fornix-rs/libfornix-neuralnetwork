use super::*;

/// Describes a layer in the neural network
/// Manages neurons within that layer
pub struct Layer {
    /// Neurons that are contained
    pub neurons: Vec<Neuron>,
}

impl Layer {
    /// Creates an empty layer
    pub fn new() -> Self {
        Layer {
            neurons: Vec::new(),
        }
    }
}

/// Describes a neuron
pub struct Neuron {
    /// Neuron model, describes the memory model and functional model
    pub model: Box<NeuronModel>,
    /// List of connections where the neuron is connected to
    pub connections: Vec<Connection>,
}

impl Neuron {
    /// Creates a new neuron without any connections
    pub fn new<T: NeuronModel + 'static>(model: T) -> Self {
        Neuron {
            model: Box::new(model),
            connections: Vec::new(),
        }
    }
}

/// Describes a connection from a neuron to an second neuron
pub struct Connection {
    /// Connection weight
    pub weight: f64,
    /// Link/Target (layer, neuron)
    pub link: (usize, usize),
}

impl Connection {
    /// Quick initializer: creates a connection object with a weight and target
    pub fn new(weight: f64, link: (usize, usize)) -> Self {
        Connection {
            weight: weight,
            link: link,
        }
    }
}

/// A neural network is form of an intelligent program.
/// As every intelligent program it takes inputs and gives outputs
///
/// The neural network is designed after the human brain in very abstract way.
/// It's one of the simplest form and is easy to train.
pub struct NeuralNetwork {
    /// Container for all layers
    pub layers: Vec<Layer>,
}

impl NeuralNetwork {
    /*
    pub fn new() -> Self {

    }
    */

    /// Gets a mutable reference to an neuron by specifying its location (layer_index, neuron_index)
    /// Returns `None` if location is out of range, see logs for detailed error message
    pub fn locate_mut(&mut self, location: (usize, usize)) -> Option<&mut Neuron> {
        if location.0 >= self.layers.len() {
            error!("tried to access a layer that is out of range");
            return None;
        }

        if location.1 >= self.layers[location.0].neurons.len() {
            error!("tried to access a neuron that is out of range");
            return None;
        }

        Some(&mut self.layers[location.0].neurons[location.1])
    }

    /// Gets a non-mutable reference to an neuron by specifying its location (layer_index, neuron_index)
    /// Returns `None` if location is out of range, see logs for detailed error message
    pub fn locate(&self, location: (usize, usize)) -> Option<&Neuron> {
        if location.0 >= self.layers.len() {
            error!("tried to access a layer that is out of range");
            return None;
        }

        if location.1 >= self.layers[location.0].neurons.len() {
            error!("tried to access a neuron that is out of range");
            return None;
        }

        Some(&self.layers[location.0].neurons[location.1])
    }
}
