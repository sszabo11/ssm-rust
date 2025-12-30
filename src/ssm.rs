use std::{
    fmt::write,
    fs::File,
    io::{BufWriter, Write},
};

use ndarray::{Array1, Array2, s};
use rand::Rng;

use crate::embedding::Embedding;

#[allow(non_snake_case, clippy::upper_case_acronyms)]
pub struct SSM {
    A: Array2<f32>,
    B: Array2<f32>,
    C: Array2<f32>,
    D: Array2<f32>,
    hidden: Array1<f32>,

    gradient_C: Array2<f32>,
    gradient_B: Array2<f32>,
}

fn init_a(d_state: usize) -> Array2<f32> {
    let mut a = Array2::zeros((d_state, d_state));
    for i in 0..d_state {
        a[[i, i]] = -(i as f32 + 1.0).ln(); // or -(0.5 + i as f32)
    }
    a
}

impl SSM {
    pub fn new(d_model: usize, d_state: usize) -> Self {
        let mut rng = rand::rng();

        let a = Array2::from_shape_fn((d_state, d_state), |_| rng.random::<f32>() * 0.02 - 0.01);
        let b = Array2::from_shape_fn((d_state, d_model), |_| rng.random::<f32>() * 0.02 - 0.05);
        let c = Array2::from_shape_fn((d_model, d_state), |_| rng.random::<f32>() * 0.02 - 0.05);

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

    pub fn forward_vec(&mut self, input: Array1<f32>) -> Array1<f32> {
        //println!("Hidden BEFORE: {:?}", &self.hidden.slice(s![..3]));
        //let x = Array1::from_vec(vec![input]);
        // How state changes
        let a_h = self.A.dot(&self.hidden);

        // How input influences the state
        let b_x = self.B.dot(&input);

        //println!("A*h: {:?}", &a_h.slice(s![..3]));
        //println!("B*x: {:?}", &b_x.slice(s![..3]));
        // Update hidden -> Beocomes compressed memory
        self.hidden = a_h + b_x;

        //println!("Hidden AFTER: {:?}", &self.hidden.slice(s![..3]));

        let output = self.C.dot(&self.hidden);
        //println!("Output: {:?}", &output.slice(s![..3]));
        output
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

    pub fn predict_sequence(&mut self, start: &[Array1<f32>], n_predict: usize) -> Vec<Vec<f32>> {
        self.reset_state();

        let mut result = Vec::new();

        for val in start {
            self.forward_vec(val.clone());
        }

        let mut last_output = start.iter().last().unwrap().clone();
        for _ in 0..n_predict {
            let pred = self.forward_vec(last_output);
            result.push(pred.to_vec());
            last_output = pred;
        }

        result
    }

    pub fn train_full(&mut self, tr: &Vec<Vec<String>>, lr: f32, model: &mut Embedding) {
        //self.reset_state();
        //self.reset_gradients();
        let epochs = 90;
        for epoch in 1..=epochs {
            let mut total_loss: f32 = 0.0;
            for sentence in tr {
                let input_embs: Vec<Array1<f32>> = sentence[..sentence.len() - 1]
                    .iter()
                    .map(|w| model.encode_word(w))
                    .collect::<Vec<Array1<f32>>>();

                let target_embs: Vec<Array1<f32>> =
                    sentence[1..].iter().map(|w| model.encode_word(w)).collect();

                let loss = self.train(&input_embs, &target_embs, lr);

                total_loss += loss;
            }

            if total_loss.is_nan() {
                break;
            }

            //if epoch == epochs {
            let avg_loss = total_loss / tr.len() as f32;
            println!("Epoch: {} | Loss: {}", epoch, avg_loss);
            //}
        }
    }

    pub fn save_weights(&self, path: &str) -> std::io::Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        let mut c: Vec<_> = self.gradient_C.iter().collect();
        let mut b: Vec<_> = self.gradient_B.iter().collect();

        for c in c.iter() {
            write(&mut c.to_string(), format_args!("")).unwrap();
        }
        for b in b.iter() {
            write(&mut b.to_string(), format_args!("")).unwrap();
        }

        Ok(())
    }

    pub fn train(&mut self, inputs: &[Array1<f32>], targets: &[Array1<f32>], lr: f32) -> f32 {
        self.reset_state();
        self.reset_gradients();
        let mut total_loss = 0.0;

        let mut outputs = Vec::new();
        let mut states = Vec::new();

        for input in inputs {
            states.push(self.hidden.clone());
            let output = self.forward_vec(input.clone());
            //println!("out: {}", output);

            outputs.push(output);
        }

        for (i, (output, target)) in outputs.iter().zip(targets.iter()).enumerate() {
            let error = output - target;

            let loss: f32 = error.iter().map(|e| e.powi(2)).sum();

            total_loss += loss;

            let state = &states[i];
            for row in 0..self.C.nrows() {
                for col in 0..self.C.ncols() {
                    let grad = 2.0 * error[row] * state[col];
                    self.gradient_C[[row, col]] += grad.clamp(-10.0, 10.0);
                    //self.gradient_C[[row, col]] += grad;
                    //self.gradient_C[[row, col]] += 2.0 * error * state[col]
                }
            }
            //if i < inputs.len() {
            //    let input_val = inputs[i];
            //    for row in 0..self.B.nrows() {
            //        for col in 0..self.B.ncols() {
            //            self.gradient_B[[row, col]] += 2.0 * error * input_val * 0.1;
            //        }
            //    }
            //}
        }

        //let max_grad = 10.0;
        //for grad in self.gradient_C.iter_mut() {
        //    *grad = grad.clamp(-max_grad, max_grad);
        //}

        let n = targets.len() as f32;

        self.C -= &(&self.gradient_C * (lr / n));
        //self.B -= &(&self.gradient_B * (lr / n));

        total_loss / n
    }

    pub fn load_weights(path: &str) {}

    pub fn reset_state(&mut self) {
        self.hidden.fill(0.0);
    }

    pub fn reset_gradients(&mut self) {
        self.gradient_C.fill(0.0);
        self.gradient_B.fill(0.0);
    }
}
