use std::{collections::HashMap, f64};

use ndarray::{Array, Array1, Array2, ArrayView, ArrayView1, Ix1, Ix2, ShapeArg};
use ndarray_rand::RandomExt;
use rand::distr::{Distribution, Uniform, weighted::WeightedIndex};

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

#[derive(Debug)]
pub struct ForwardOutput {
    pub h_t: Array1<f32>,
    pub z_t: Array1<f32>,
    //pub a_t: Array1<f32>,
    pub x_t: Array1<f32>,
    //pub target_y: Array1<f32>,
    //pub pred_y: Array1<f32>,
    pub target_idx: usize,
    pub logits_t: Array1<f32>,
}

impl RNN {
    pub fn new(hidden_size: usize, seq_length: usize, corpus: &str, embedding_dim: usize) -> Self {
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
            h: Array1::zeros(hidden_size),

            // Output
            W_hy: Array2::random(
                (vocab_size, hidden_size), // A matrix with embedding dim rows and hidden cols
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
            by: Array1::zeros(vocab_size),
        }
    }

    pub fn forward(&mut self, input: &Array1<usize>) -> Vec<ForwardOutput> {
        let seq_len = input.len();
        let mut output: Vec<ForwardOutput> = Vec::with_capacity(input.len());

        for t in 0..(seq_len - 1) {
            println!("{} {}", t, seq_len);
            let curr_idx = input[t];
            let next_idx = input[t + 1];

            let x = self.W_emb.row(curr_idx);

            // Pre-Activation (logits or linear output)
            let z_t = self.W_hh.dot(&self.h) + self.W_xh.dot(&x) + &self.bh;

            let h_t = z_t.mapv(sigmoid);
            self.h = h_t.clone();

            // Output logits
            let logits_t = self.W_hy.dot(&self.h) + &self.by;

            output.push(ForwardOutput {
                h_t: h_t.clone(),
                z_t,
                x_t: x.to_owned(),
                logits_t,
                target_idx: next_idx,
            });
        }
        output
    }

    // Cross entropy
    pub fn loss(&mut self, outputs: &[ForwardOutput], input: Array1<usize>) -> f32 {
        let target_idxs = &input.to_vec()[1..];

        let mut total = 0.0;

        for (out, &target_idx) in outputs.iter().zip(target_idxs) {
            total += self.cross_entropy_with_logits(&out.logits_t, target_idx)
        }

        total / outputs.len() as f32
    }

    pub fn inference(&mut self, seed: &str, len: usize) -> String {
        let mut response = String::new();

        for char_idx in seed
            .chars()
            .map(|c| self.char_to_i.get(&c).unwrap())
            .collect::<Vec<&usize>>()
        {
            let e = self.W_emb.row(*char_idx);
            let z_t = self.W_hh.dot(&self.h) + self.W_xh.dot(&e) + &self.bh;

            let a_t = z_t.mapv(sigmoid);

            self.h = a_t;
        }

        while response.len() < len {
            let logits = self.W_hy.dot(&self.h) + &self.by;

            let probs = softmax(&logits);

            //let next_idx = self.sample_categorical(&probs);

            let next_idx = argmax(&logits);

            //println!("enxt idx: {}", next_idx);
            let char = self.i_to_char.get(&next_idx).unwrap();
            response.push(*char);

            let x_emb = self.W_emb.row(next_idx);
            let z_t = self.W_hh.dot(&self.h) + self.W_xh.dot(&x_emb) + &self.bh;

            let a_t = z_t.mapv(sigmoid);

            self.h = a_t;
        }

        response
    }

    pub fn cross_entropy(&self, probs: &Array1<f32>, target_idx: usize) -> f32 {
        -probs[target_idx].ln().max(-100.0) // clip to avoid log(0) = -inf
    }

    fn cross_entropy_with_logits(&self, logits: &Array1<f32>, correct_idx: usize) -> f32 {
        // Numerical stability: subtract max logit
        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let log_sum_exp = (logits.mapv(|v| (v - max_logit).exp())).sum().ln() + max_logit;
        -(logits[correct_idx] - log_sum_exp)
    }

    pub fn train(
        &mut self,
        input: Array1<usize>, // Sentence of chars as idxs to embedding
        batch_size: usize,
        lr: f32,
    ) -> Option<()> {
        let mut total_gradient_W_xh = Array2::<f32>::zeros((self.hidden_size, self.embedding_dim));
        let mut total_gradient_W_hh = Array2::<f32>::zeros((self.hidden_size, self.hidden_size));
        let mut total_gradient_W_hy = Array2::<f32>::zeros((self.vocab_size, self.hidden_size));

        //let mut total_gradient_by = Array1::zeros(self.vocab_size);
        //let mut total_gradient_bh = Array1::zeros(self.hidden_size);

        self.clear_hidden();

        if input.len() == 0 {
            return None;
        };

        let forward_outputs = self.forward(&input);

        let loss = self.loss(&forward_outputs, input);
        println!("Loss: {}", loss);

        let mut delta_h_next = Array1::zeros(self.hidden_size);

        // Backprop
        for t in (0..forward_outputs.len()).rev() {
            //let y_t = &forward_outputs[t].pred_y; // Predicted Output
            let z_t = &forward_outputs[t].z_t; // Pre-Activation
            let h_t = &forward_outputs[t].h_t; // Hidden state
            let target_idx = &forward_outputs[t].target_idx;
            let logits_t = &forward_outputs[t].logits_t;
            let h_prev = if t == 0 {
                h_t
            } else {
                &forward_outputs[t - 1].h_t
            }; // Hidden t-1 state
            //let a_t = &pred_a_vec[t]; // Activation state
            let x_t = &forward_outputs[t].x_t; // Input state
            //let target_y = &forward_outputs[t].target_y; // Real output state

            let sig_der = z_t.map(|z| sigmoid_der(*z)); // Sigmoid derivative element wise
            //let delta_y = y_t - target_y; // derivative dL/da(L) or dL/dy

            //loss = delta_y.sum() / delta_y.len() as f32;
            let probs = softmax(logits_t);
            let mut delta_y = probs.clone();
            *delta_y.get_mut(*target_idx).unwrap() -= 1.0;

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
        //}
        Some(())
    }

    pub fn clear_hidden(&mut self) {
        self.h = Array1::zeros(self.h.dim());
    }
}

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
//fn cross_entropy_loss(logits: &Array1<f32>, target_idx: usize) -> f32 {
//    let probs = softmax(logits); // you implement softmax
//    -probs[target_idx].ln() // negative log prob of true class
//}

fn sample(probs: Array1<f32>) -> usize {
    println!("probs dim: {}", probs.dim());
    probs.to_vec().sort_by(|a, b| b.total_cmp(a));

    2
}

pub fn argmax(arr: &Array1<f32>) -> usize {
    arr.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

//fn outer_product<'a>(x: &'a Array1<f32>, y: &'a Array1<f32>) -> Array2<f32> {
//    a.dot(&b.t())
//}
