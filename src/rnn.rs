use std::{collections::HashMap, f64};

use ndarray::{Array, Array1, Array2, ArrayView, ArrayView1, Ix1, Ix2, ShapeArg};
use ndarray_rand::RandomExt;
use rand::distr::Uniform;

use crate::embedding2::{EmbeddingChar, get_chars};

pub struct RNN {
    #[allow(non_snake_case)]
    W_hh: Array2<f32>, // Weight for hidden - hidden
    #[allow(non_snake_case)]
    W_xh: Array2<f32>, // Weight for input - hidden
    #[allow(non_snake_case)]
    W_hy: Array2<f32>, // Weight for hidden - output

    #[allow(non_snake_case)]
    pub W_emb: Array2<f32>, // Weight for embedding

    by: Array1<f32>,
    bh: Array1<f32>,

    h: Array1<f32>,
    hidden_size: usize,
    embedding_dim: usize,
    seq_length: usize,
    //pub embedding: EmbeddingChar,
    pub vocab_size: usize,
    pub char_to_i: HashMap<char, usize>,
    pub i_to_char: HashMap<usize, char>,
}

pub struct ForwardOutput {
    pub h_t: Array1<f32>,
    pub z_t: Array1<f32>,
    pub a_t: Array1<f32>,
    pub x_t: Array1<f32>,
    pub target_y: Array1<f32>,
    pub pred_y: Array1<f32>,
}

impl RNN {
    pub fn new(
        hidden_size: usize,
        seq_length: usize,
        corpus: &str,
        embedding_dim: usize,
        sliding_window: usize,
        k: usize,
    ) -> Self {
        let vocab_size = get_chars(corpus, 1).len();
        let mut char_to_i: HashMap<char, usize> = HashMap::new();
        let mut i_to_char: HashMap<usize, char> = HashMap::new();

        let mut unique_chars: Vec<char> = corpus.chars().collect();
        unique_chars.sort_unstable();
        unique_chars.dedup();

        for (idx, c) in unique_chars.into_iter().enumerate() {
            char_to_i.insert(c, idx);
            i_to_char.insert(idx, c);
        }

        Self {
            vocab_size,
            hidden_size,
            seq_length,
            char_to_i,
            i_to_char,
            embedding_dim,
            //embedding: EmbeddingChar::new(corpus.to_string(), embedding_dim, sliding_window, k),
            h: Array1::zeros(hidden_size),

            // Output
            W_hy: Array2::random(
                (embedding_dim, hidden_size), // A matrix with embedding dim rows and hidden cols
                Uniform::new(-0.1, 0.1).unwrap(),
            ),
            W_emb: Array2::random(
                (vocab_size, embedding_dim), // A matrix with embedding dim rows and hidden cols
                Uniform::new(-0.1, 0.1).unwrap(),
            ),
            // Input
            W_xh: Array2::random(
                (hidden_size, embedding_dim), // A matrix with hidden rows and embedding dim cols
                Uniform::new(-0.1, 0.1).unwrap(),
            ),
            W_hh: Array2::random((hidden_size, hidden_size), Uniform::new(-0.1, 0.1).unwrap()),
            bh: Array1::zeros(hidden_size),
            by: Array1::zeros(embedding_dim),
        }
    }

    pub fn forward(&mut self, input: Array1<usize>) -> Vec<ForwardOutput> {
        let mut output: Vec<ForwardOutput> = Vec::with_capacity(input.len());

        for idx in input {
            let x = self.W_emb.row(idx);
            //println!("x {}", x.dim());
            //println!("y {}", y_idx);
            //

            let y_idx = idx + 1;
            let target_y_embedding = self.W_emb.row(y_idx);

            // Pre-Activation (logits or linear output)
            let z_t = self.W_hh.dot(&self.h) + self.W_xh.dot(&x) + &self.bh;

            // Hidden state (This is activation a(L))

            //let a_t = z_t.tanh();
            let a_t = z_t.mapv(sigmoid);

            self.h = a_t.clone();

            // Prediction / output
            let pred_y = self.W_hy.dot(&self.h) + &self.by;

            output.push(ForwardOutput {
                h_t: a_t.clone(),
                z_t,
                pred_y,
                a_t,
                x_t: x.to_owned(),
                target_y: target_y_embedding.to_owned(),
            });
        }
        output
    }

    // MSE
    // First part of the chain
    pub fn loss(
        &mut self,
        target_y: ArrayView1<f32>,
        predicted_y: ArrayView1<f32>,
        n: usize,
    ) -> Array1<f32> {
        // dL/dyt
        (2.0 * (&target_y - &predicted_y)) / n as f32
    }

    pub fn backward(&mut self, loss: ArrayView1<f32>) -> Array1<f32> {
        //self.W_xh.map(|w| {
        //    //let loss = self.loss(target_y, predicted_y, n);
        //});
        //
        Array1::zeros(self.hidden_size)
    }

    pub fn inference(&mut self, seed: &str, len: usize) -> String {
        let mut response = String::new();

        let last_char = seed.chars().last().unwrap();
        let idx = self.char_to_i.get(&last_char).expect("Char not found");

        while response.len() <= len {
            let x = self.W_emb.row(*idx);

            let z_t = self.W_hh.dot(&self.h) + self.W_xh.dot(&x) + &self.bh;

            //let a_t = z_t.tanh();
            let a_t = z_t.map(|z| sigmoid(*z));

            self.h = a_t.clone();

            // Prediction / output
            let pred_y = self.W_hy.dot(&self.h) + &self.by;

            println!("{}", pred_y.dim());

            //for idx in pred_y.iter() {
            //    println!("{}", self.i_to_char.get(&(*idx as usize)).unwrap());
            //}

            let c = self.i_to_char.get(&(pred_y[0] as usize)).unwrap();
            response.push(*c);
        }
        //let final_logits = &outputs.last().unwrap();
        //let probs = softmax(final_logits); // now probs sum to 1
        //let next_char_idx = sample_from_probs(&probs); // e.g., weighted random or argmax
        //let next_char = idx_to_char[next_char_idx];

        response
    }

    pub fn train(
        &mut self,
        input: Array1<usize>, // Sentence of chars as idxs to embedding
        //target: Array2<f32>,
        batch_size: usize,
        lr: f32,
    ) {
        let mut loss = 0.0;
        println!("{}", input.len());
        // x = idx of char of embedding
        for (i, x) in input.clone().into_iter().enumerate() {
            if i >= input.len() - 1 {
                continue;
            }

            println!("Loss: {}", loss);
            //println!("i: {} | L: {}", i, input.len());
            let target = input[i + 1];
            // next idx of char of embedding
            let target_y = input.get(target).unwrap_or(&0);

            let target_y_embedding = self.W_emb.row(*target_y);

            // Cache the predictions for backprop
            let mut pred_y_vec = Vec::new(); // Outputs of previous timesteps
            let mut pred_z_vec = Vec::new(); // Pre-Activation of previous timesteps
            let mut pred_h_vec = Vec::new(); // Hidden state of previous timesteps
            let mut pred_a_vec = Vec::new(); // Activation of previous timesteps
            let mut pred_x_vec = Vec::new(); // Inputs of previous timesteps
            let mut pred_real_y_vec = Vec::new(); // Target outputs previous timesteps
            //let mut pred_L_vec = Vec::new(); // Losses of previous timesteps

            let mut total_gradient_W_xh =
                Array2::<f32>::zeros((self.hidden_size, self.embedding_dim));
            let mut total_gradient_W_hh =
                Array2::<f32>::zeros((self.hidden_size, self.hidden_size));
            let mut total_gradient_W_hy =
                Array2::<f32>::zeros((self.embedding_dim, self.hidden_size));

            //let mut total_gradient_by = Array1::zeros(self.vocab_size);
            //let mut total_gradient_bh = Array1::zeros(self.hidden_size);

            // Forward
            //for t in 0..input.len() {
            self.clear_hidden();

            //if i >= input.len() {
            //    continue 'forward;
            //}
            let f_o = self.forward(input);

            //let l = (f_o.pred_y - target_y_embedding).pow2();
            //pred_L_vec.push(loss);

            // Cache for back pass
            pred_y_vec.push(f_o.pred_y);
            pred_z_vec.push(f_o.z_t);

            // These are the same!! TODO: FIX
            pred_h_vec.push(f_o.h_t);
            pred_a_vec.push(f_o.a_t);

            pred_x_vec.push(f_o.x_t);
            pred_real_y_vec.push(f_o.target_y);
            //}
            let mut delta_h_next = Array1::zeros(self.hidden_size);

            // Backprop
            for t in (0..input.len()).rev() {
                let y_t = &pred_y_vec[t]; // Predicted Output
                let z_t = &pred_z_vec[t]; // Pre-Activation
                let h_t = &pred_h_vec[t]; // Hidden state
                let h_prev = if t == 0 { h_t } else { &pred_h_vec[t - 1] }; // Hidden t-1 state
                //let a_t = &pred_a_vec[t]; // Activation state
                let x_t = &pred_x_vec[t]; // Input state
                let target_y = &pred_real_y_vec[t]; // Real output state

                let sig_der = z_t.map(|z| sigmoid_der(*z)); // Sigmoid derivative element wise
                let delta_y = y_t - target_y; // derivative dL/da(L) or dL/dy

                loss = delta_y.sum() / delta_y.len() as f32;
                let delta_h = self.W_hy.t().dot(&delta_y) + &delta_h_next;
                let delta_z = &delta_h * sig_der; // derivative dL/da(L) or dL/dy

                {
                    // THIS uses x_t
                    // For w_xh
                    let gradient_xh = outer_product(&delta_z, x_t);
                    total_gradient_W_xh += &gradient_xh;
                }

                {
                    // THIS USES h{t-1}
                    // For w_hh
                    let gradient_hh = outer_product(&delta_z, h_prev);
                    total_gradient_W_hh += &gradient_hh;
                }
                {
                    // THIS USES h_t
                    // For w_hy
                    let gradient_hy = outer_product(&delta_y, h_t);

                    total_gradient_W_hy += &gradient_hy;
                    //let total_W_hy = &delta_y * &der_z_wrt_w * &der_a_wrt_z;
                }
                delta_h_next = self.W_hh.t().dot(&delta_z);
                // Propagate error to hidden state

                //let w_hy_T = self.W_hy.to_shape((self.W_hy.nrows(), 1)).unwrap();
                //let delta_h_t = w_hy_T.dot(&delta_y_t) + delta_h_t1;

                //let loss = self.loss(target_y_embedding.t(), pred_y.t(), batch_size);
            }

            self.W_xh -= &(lr * total_gradient_W_xh);
            self.W_hh -= &(lr * total_gradient_W_hh);
            self.W_hy -= &(lr * total_gradient_W_hy);

            //self.bh -= &(lr * total_gradient_bh);
            //self.by -= &(lr * total_gradient_by);
        }
    }

    fn clear_hidden(&mut self) {
        self.h = Array1::zeros(self.h.dim());
    }
}

//fn outer<D: ShapeArg>(a: &Array<f32, D>, b: &Array<f32, D>) -> &'static Array<f32, D> {
//    let (size_a, size_b) = (a.shape()[0], b.shape()[0]);
//
//    let a_t = a.clone().to_shape((size_a, 1)).unwrap();
//    let b_t = b.clone().to_shape((size_b, 1)).unwrap();
//
//    a_t.dot(b_t)
//}

//fn calculate_weight(target_y: Array1<f32>, pred_y: Array1<f32>) -> Array1<f32> {}

pub fn outer_product(x: &Array<f32, Ix1>, y: &Array<f32, Ix1>) -> Array<f32, Ix2> {
    let (size_x, size_y) = (x.shape()[0], y.shape()[0]);
    let x_view = x.view();
    let y_view = y.view();
    let x_reshaped = x_view.to_shape((size_x, 1)).unwrap();
    let y_reshaped = y_view.to_shape((1, size_y)).unwrap();
    x_reshaped.dot(&y_reshaped)
}

fn softmax(nums: &Array1<f32>) -> Array1<f32> {
    let sum: f64 = nums.iter().map(|n| f64::consts::E.powf(*n as f64)).sum();

    nums.iter()
        .map(|n| (f64::consts::E.powf(*n as f64) / sum) as f32)
        .collect::<Array1<f32>>()
}

fn sigmoid_der(x: f32) -> f32 {
    sigmoid(x) * (1.0 - sigmoid(x))
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn test_softmax() {
        let nums = array![-1.0, 0.0, 3.0, 5.0];

        let soft = softmax(&nums);

        assert_eq!(soft.iter().sum::<f32>(), 1.0);
        assert_eq!(soft[0], 0.00216569646);
    }

    #[test]
    fn test_outer() {
        let a = array![1., 2., 3.];
        let b = array![4., 5., 6.];

        let outer = &outer_product(&a, &b);

        println!("{}", outer);
        //assert_eq!(outer, array![32]);
    }
}
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
fn cross_entropy_loss(logits: &Array1<f32>, target_idx: usize) -> f32 {
    let probs = softmax(logits); // you implement softmax
    -probs[target_idx].ln() // negative log prob of true class
}

//fn outer_product<'a>(x: &'a Array1<f32>, y: &'a Array1<f32>) -> Array2<f32> {
//    a.dot(&b.t())
//}
