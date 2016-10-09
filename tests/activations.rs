extern crate libfornix_neuralnetwork;

use libfornix_neuralnetwork::*;

#[test]
fn test_arctan() {
    let inputs = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let outputs = ArcTanActivation::activate(inputs);
}
