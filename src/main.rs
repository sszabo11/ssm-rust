use std::collections::HashMap;

use ndarray::{Array, Array1, Array2, array};
use rand::Rng;

#[allow(non_snake_case, clippy::upper_case_acronyms)]
struct SSM {
    A: Array2<f32>,
    B: Array2<f32>,
    C: Array2<f32>,
    D: Array2<f32>,
    hidden: Array1<f32>,

    gradient_C: Array2<f32>,
    gradient_B: Array2<f32>,
}

impl SSM {
    pub fn new(d_model: usize, d_state: usize) -> Self {
        let mut rng = rand::rng();

        let a = Array2::from_shape_fn((d_state, d_state), |_| rng.random::<f32>() * 0.02 - 0.01);
        let b = Array2::from_shape_fn((d_state, d_model), |_| rng.random::<f32>() * 0.02 - 0.1);
        let c = Array2::from_shape_fn((d_model, d_state), |_| rng.random::<f32>() * 0.02 - 0.1);

        Self {
            A: a.clone(),
            B: b.clone(),
            C: c.clone(),
            D: Array2::zeros((0, 0)),
            gradient_C: Array2::zeros(c.dim()),
            gradient_B: Array2::zeros(b.dim()),
            hidden: Array1::zeros(d_state),
        }
    }

    pub fn run(&mut self, input: Array1<f32>) -> Array1<f32> {
        // How state changes
        let a_h = self.A.dot(&self.hidden);

        // How input influences the state
        let b_x = self.B.dot(&input);

        // Update hidden -> Beocomes compressed memory
        self.hidden = a_h + b_x;

        // Output
        self.C.dot(&self.hidden)
    }

    pub fn forward(&mut self, input: f32) -> f32 {
        let x = Array1::from_vec(vec![input]);
        // How state changes
        let a_h = self.A.dot(&self.hidden);

        // How input influences the state
        let b_x = self.B.dot(&x);

        // Update hidden -> Beocomes compressed memory
        self.hidden = a_h + b_x;

        self.C.dot(&self.hidden)[0]
    }

    pub fn forward_all(&mut self, input: &[f32]) -> Array1<f32> {
        // How state changes
        let a_h = self.A.dot(&self.hidden);

        // How input influences the state
        let b_x = self.B.dot(&Array1::from_vec(input.to_vec()));

        // Update hidden -> Beocomes compressed memory
        self.hidden = a_h + b_x;

        self.C.dot(&self.hidden)
    }

    pub fn predict_sequence(&mut self, start: &[f32], n_predict: usize) -> Vec<f32> {
        self.reset_state();

        let mut result = Vec::new();

        for &val in start {
            self.forward(val);
        }

        let mut last_output = start.last().copied().unwrap_or(0.0);
        for _ in 0..n_predict {
            let pred = self.forward(last_output);
            result.push(pred);
            last_output = pred;
        }

        result
    }

    pub fn train(&mut self, inputs: &[f32], targets: &[f32], lr: f32) -> f32 {
        self.reset_state();
        self.reset_gradients();
        let mut total_loss = 0.0;

        let mut outputs = Vec::new();
        let mut states = Vec::new();

        for &input in inputs {
            states.push(self.hidden.clone());
            let output = self.forward(input);

            outputs.push(output);
        }

        for (i, (output, target)) in outputs.iter().zip(targets.iter()).enumerate() {
            let error = output - target;

            let loss = error * error;

            total_loss += loss;

            let state = &states[i];
            for row in 0..self.C.nrows() {
                for col in 0..self.C.ncols() {
                    self.gradient_C[[row, col]] += 2.0 * error * state[col]
                }
            }

            if i < inputs.len() {
                let input_val = inputs[i];
                for row in 0..self.B.nrows() {
                    for col in 0..self.B.ncols() {
                        self.gradient_B[[row, col]] += 2.0 * error * input_val * 0.1;
                    }
                }
            }
        }
        let n = targets.len() as f32;

        self.C -= &(&self.gradient_C * (lr / n));
        self.B -= &(&self.gradient_B * (lr / n));

        total_loss / n
        //self.forward_all(input);
        //for i in 0..rows {
        //    for j in 0..cols {
        //        for (i, n) in expected_output.iter().enumerate() {
        //            // Output
        //            let prediction = self.C.dot(&self.hidden)[i];
        //            let target = expected_output[j];
        //            self.gradient_C.row_mut(i)[j] = (prediction - target) * self.hidden[j];
        //        }
        //    }
        //}

        //self.C = &self.C - lr * &self.gradient_C;
    }

    pub fn reset_state(&mut self) {
        self.hidden.fill(0.0);
    }

    fn reset_gradients(&mut self) {
        self.gradient_C.fill(0.0);
        self.gradient_B.fill(0.0);
    }
}

fn main() {
    let mut ssm = SSM::new(1, 24);

    let training_sequences = vec![
        // Counting by 1s
        (vec![0.0, 1.0, 2.0, 3.0], vec![1.0, 2.0, 3.0, 4.0]),
        (vec![5.0, 6.0, 7.0, 8.0], vec![6.0, 7.0, 8.0, 9.0]),
        (vec![10.0, 11.0, 12.0, 13.0], vec![11.0, 12.0, 13.0, 14.0]),
        // Counting by 2s
        (vec![0.0, 2.0, 4.0, 6.0], vec![2.0, 4.0, 6.0, 8.0]),
        (vec![1.0, 3.0, 5.0, 7.0], vec![3.0, 5.0, 7.0, 9.0]),
        // Counting by 5s
        (vec![0.0, 5.0, 10.0, 15.0], vec![5.0, 10.0, 15.0, 20.0]),
    ];

    let mut curr = 0.000001;
    let end = 0.001;

    let mut result = HashMap::new();
    while curr < end {
        ssm.reset_state();
        ssm.reset_gradients();
        let epochs = 5000;
        for epoch in 1..=epochs {
            let mut total_loss = 0.0;
            for (input, target) in &training_sequences {
                let loss = ssm.train(input, target, curr);

                total_loss += loss;
            }

            if epoch == 5000 {
                println!("Epoch: {} | Loss: {}", epoch, total_loss);
                result.insert(total_loss.to_string(), curr);
            }
        }
        curr += 0.000001;
    }

    let mut losses = result
        .keys()
        .cloned()
        .map(|k| k.parse::<f32>().unwrap())
        .collect::<Vec<f32>>();

    losses.sort_by(f32::total_cmp);

    println!(
        "Best: {:?} @ lr: {}",
        losses[0],
        result.get(&losses[0].to_string()).unwrap()
    );

    let out1 = ssm.predict_sequence(&[0.0, 1.0, 2.0, 3.0], 4);
    println!();
    println!("Expected: {:?}", &[4.0, 5.0, 6.0, 7.0]);
    println!("Result: {:?}", out1);
    println!();

    let out2 = ssm.predict_sequence(&[3.0, 5.0, 7.0, 9.0], 4);
    println!();
    println!("Expected: {:?}", &[11.0, 13.0, 15.0, 17.0]);
    println!("Result: {:?}", out2);
    println!();

    let out3 = ssm.predict_sequence(&[10.0, 20.0, 30.0], 3);
    println!();
    println!("Expected: {:?}", &[40.0, 50.0, 60.0]);
    println!("Result: {:?}", out3);
    println!();

    println!();
    let out4 = ssm.predict_sequence(&[10.0, 30.0, 50.0], 6);
    println!("Result: {:?}", out4);
    println!();

    //let epochs = 500;
    //for epoch in 1..=epochs {
    //    let input = array![
    //        3.0 + 2.0 * (epoch as f32 * 0.1).sin(),
    //        5.0 + 1.0 * (epoch as f32 * 0.1).cos(),
    //    ];
    //    let output = ssm.run(input);
    //    println!(
    //        "Epoch {}: state={:?}, output={:?}",
    //        epoch, ssm.hidden, output
    //    );
    //}
}
