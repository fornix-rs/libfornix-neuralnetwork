use libfornix::*;

/// An Neuron turns x inputs into y outputs.
/// How these inputs are turned into outputs will be defined by this trait.
pub trait NeuronModel {
    // Creates the initial/first generation of an neuron.
    // Usually this is a neuron with completely random values
    //fn create_initial(rng: &mut RandomNumberProvider) -> Self;

    /// Calculates the neuron value.
    /// This will be the value that will be forwarded to the next layer
    /// `inputs` - a vector of *activated* inputs
    fn calculate(&self, inputs: &Vec<f64>) -> f64;

    /// Calculates the neuron value.
    /// This will be the value that will be forwarded to the next layer
    /// Compared to `calculate`this allows the neuron model to also adjust it self
    /// This is useful for neuron models that have memory for example.
    /// The default implementation will simply call the non-mutable function
    fn calculate_mut(&mut self, inputs: &Vec<f64>) -> f64 {
        self.calculate(inputs)
    }
}

/// A trainable neuron model is able to be adjusted
/// It allows to change values that are constant during runtime
pub trait TrainableNeuronModel {
    /// Returns a reference to vector for the tweakable values.
    /// The first value is the value it self, the second tuple is the range the values can be.
    /// This is essential for the training process. The training host will be able to adjust these values
    fn tweakable_values<'a>(&'a mut self) -> Vec<(&'a mut f64, (f64, f64))>;
}

/// The TrivialNeuron as its name suggests, is the simplest way of handling neurons
/// It will calculate it's value by summing up all activated input values
pub struct TrivialNeuron {
    pub bias: f64,
}

impl TrivialNeuron {
    pub fn create_initial(rng: &mut RandomNumberProvider) -> Self {
        TrivialNeuron {
            bias: rng.generate_number(-1.0, 1.0),
        }
    }
}

impl TrainableNeuronModel for TrivialNeuron {
    fn tweakable_values<'a>(&'a mut self) -> Vec<(&'a mut f64, (f64, f64))> {
        vec![
            (&mut self.bias, (-1.0, 1.0)),
        ]
    }
}

impl NeuronModel for TrivialNeuron {
    fn calculate(&self, inputs: &Vec<f64>) -> f64 {
        let mut value = 0.0;

        for num in inputs {
            value += *num;
        }

        value + self.bias
    }
}

/// A Input neuron is used in the first layer, it will hold on value
pub struct InputNeuron {
   pub value: f64,
}

impl NeuronModel for InputNeuron {
    fn calculate(&self, inputs: &Vec<f64>) -> f64 {
        self.value
    }
}

impl TrainableNeuronModel for InputNeuron {
    fn tweakable_values<'a>(&'a mut self) -> Vec<(&'a mut f64, (f64, f64))> {
        Vec::new()
    }
}

/// An Output neuron is used in the last layer does not hold anything.
/// It will only summarize its inputs
pub struct OutputNeuron;

impl NeuronModel for OutputNeuron {
    fn calculate(&self, inputs: &Vec<f64>) -> f64 {
        let mut value = 0.0;

        for num in inputs {
            value += *num;
        }

        value
    }
}

impl TrainableNeuronModel for OutputNeuron {
    fn tweakable_values<'a>(&'a mut self) -> Vec<(&'a mut f64, (f64, f64))> {
        Vec::new()
    }
}
