#![allow(non_snake_case)]

use std::{
    collections::HashMap,
    f64, fs,
    io::{BufWriter, Read, Write},
};

use anyhow::Result;
use ndarray::{
    Array, Array1, Array2, Array3, ArrayView, ArrayView1, ArrayView2, Axis, Dimension, Ix1, Ix2,
    ShapeArg, ShapeBuilder,
};
use ndarray_rand::RandomExt;
use rand::{
    distr::{Distribution, Uniform, weighted::WeightedIndex},
    seq,
};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::embedding2::{EmbeddingChar, get_chars};

#[allow(clippy::upper_case_acronyms)]
pub struct RNN {
    W_hh: Array2<f32>,  // Weight for hidden - hidden
    W_xh: Array2<f32>,  // Weight for input - hidden
    W_hy: Array2<f32>,  // Weight for hidden - output
    W_emb: Array2<f32>, // Weight for embedding

    by: Array1<f32>,
    bh: Array1<f32>,

    h: Array1<f32>,
    hidden_size: usize,
    embedding_dim: usize,
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

pub struct MiniBatch {
    total_gradient_W_xh: Array2<f32>,
    total_gradient_W_hh: Array2<f32>,
    total_gradient_W_hy: Array2<f32>,
    total_gradient_by: Array1<f32>,
    total_gradient_bh: Array1<f32>,
    loss: f32,
}

impl RNN {
    pub fn new(hidden_size: usize, corpus: &str, embedding_dim: usize) -> Self {
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

    pub fn forward(&self, input: &Array1<usize>) -> Vec<ForwardOutput> {
        let seq_len = input.len();
        let mut output: Vec<ForwardOutput> = Vec::with_capacity(input.len());

        let mut h = Array1::zeros(self.hidden_size);

        for t in 0..(seq_len - 1) {
            let curr_idx = input[t];
            let next_idx = input[t + 1];

            let x = self.W_emb.row(curr_idx);

            // Pre-Activation (logits or linear output)
            let z_t = self.W_hh.dot(&h) + self.W_xh.dot(&x) + &self.bh;

            let h_t = z_t.mapv(sigmoid);
            h = h_t.clone();

            // Output logits
            let logits_t = self.W_hy.dot(&h) + &self.by;

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
    pub fn loss(&self, outputs: &[ForwardOutput], input: Array1<usize>) -> f32 {
        let target_idxs = &input.to_vec()[1..];

        let mut total = 0.0;

        for (out, &target_idx) in outputs.iter().zip(target_idxs) {
            total += self.cross_entropy_with_logits(&out.logits_t, target_idx)
        }

        total / outputs.len() as f32
    }

    pub fn inference(&mut self, seed: &str, len: usize, temp: f32) -> String {
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

            //println!("probs: {}", probs);
            let temp_probs: Array1<f32> = probs.iter().map(|v| v.powi(2) * temp).collect();

            let next_idx = sample(temp_probs);

            let char = self.i_to_char.get(&next_idx).unwrap();
            response.push(*char);

            let x_emb = self.W_emb.row(next_idx);
            let z_t = self.W_hh.dot(&self.h) + self.W_xh.dot(&x_emb) + &self.bh;

            let a_t = z_t.mapv(sigmoid);

            self.h = a_t;
        }

        response
    }

    fn cross_entropy_with_logits(&self, logits: &Array1<f32>, correct_idx: usize) -> f32 {
        // Numerical stability: subtract max logit
        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let log_sum_exp = (logits.mapv(|v| (v - max_logit).exp())).sum().ln() + max_logit;
        -(logits[correct_idx] - log_sum_exp)
    }

    pub fn train_corpus(
        &mut self,
        corpus: &str,
        epochs: usize,
        batch_size: usize,
        lr: f32,
        seq_len: usize,
    ) {
        let chars: Vec<char> = corpus.chars().collect();

        let idxs: Vec<usize> = chars
            .into_iter()
            .map(|c| self.char_to_i.get(&c).copied().expect("Char not found"))
            .collect::<Vec<usize>>();

        // Collection of sequences. Grouped in seq_len vectors
        let mut sequences: Vec<Vec<usize>> = idxs.chunks(seq_len).map(|v| v.to_vec()).collect();
        sequences.pop(); // Last one doesn't have seq length

        // Save as above but in Array
        let mut batch_input = Array3::zeros((0, batch_size, seq_len));

        let c: Vec<&[Vec<usize>]> = sequences.chunks(batch_size).collect();

        //let cc: Vec<Vec<Vec<usize>>> = c.into_iter().map(|c| c.to_vec()).collect();

        for (i, batch) in c.iter().enumerate() {
            let mut arr = Array2::zeros((0, seq_len));
            for s in batch.iter() {
                arr.push_row(Array1::from_vec(s.to_vec()).view()).unwrap();
            }
            println!("{:?} | {:?}", arr.dim(), batch_input.dim());
            if i < c.len() - 1 {
                batch_input.push(Axis(0), arr.view()).unwrap();
            }
        }
        //println!("batves: {}", batch_input);

        for epoch in 0..epochs {
            let mut loss = 0.0;

            for batch in batch_input.outer_iter() {
                let batch_loss = self.train_mini_batch(&batch, lr);
                loss = batch_loss;
            }

            //let avg_loss = loss / batch_size as f32;
            println!("Epoch: {} | Loss: {}", epoch, loss);
        }
    }

    pub fn train_mini_batch(
        &mut self,
        inputs: &ArrayView2<usize>, // Single batch
        lr: f32,
    ) -> f32 {
        let batch_size = inputs.nrows();

        let mut acc_loss = 0.0;

        let mut acc_gradient_W_xh = Array2::<f32>::zeros((self.hidden_size, self.embedding_dim));
        #[allow(non_snake_case)]
        let mut acc_gradient_W_hh = Array2::<f32>::zeros((self.hidden_size, self.hidden_size));
        #[allow(non_snake_case)]
        let mut acc_gradient_W_hy = Array2::<f32>::zeros((self.vocab_size, self.hidden_size));

        let mut acc_gradient_by = Array1::zeros(self.vocab_size);
        let mut acc_gradient_bh = Array1::zeros(self.hidden_size);

        // Each sequence in batch
        let values: Vec<MiniBatch> = (0..batch_size)
            .collect::<Vec<usize>>()
            .par_iter()
            .map(|batch| {
                let input = inputs.row(*batch).to_owned();

                #[allow(non_snake_case)]
                let mut total_gradient_W_xh =
                    Array2::<f32>::zeros((self.hidden_size, self.embedding_dim));
                #[allow(non_snake_case)]
                let mut total_gradient_W_hh =
                    Array2::<f32>::zeros((self.hidden_size, self.hidden_size));
                #[allow(non_snake_case)]
                let mut total_gradient_W_hy =
                    Array2::<f32>::zeros((self.vocab_size, self.hidden_size));

                let mut total_gradient_by = Array1::zeros(self.vocab_size);
                let mut total_gradient_bh = Array1::zeros(self.hidden_size);

                //self.clear_hidden();

                if input.is_empty() {
                    println!("em,pty");
                };

                let forward_outputs = self.forward(&input);

                let loss = self.loss(&forward_outputs, input);
                //println!("Loss: {}", loss);

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
                    };

                    let x_t = &forward_outputs[t].x_t; // Input state

                    let sig_der = z_t.map(|z| sigmoid_der(*z)); // Sigmoid derivative element wise

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

                        let gradient_hh = outer_product(&delta_y, h_t);
                        total_gradient_W_hy += &gradient_hh;
                    }

                    {
                        // THIS USES
                        // For bh

                        total_gradient_bh += &delta_z;
                    }

                    {
                        // THIS USES
                        // For by

                        total_gradient_by += &delta_y;
                    }

                    delta_h_next = self.W_hh.t().dot(&delta_z);
                }

                MiniBatch {
                    loss,
                    total_gradient_W_hy,
                    total_gradient_by,
                    total_gradient_bh,
                    total_gradient_W_xh,
                    total_gradient_W_hh,
                }
            })
            .collect();
        acc_loss += values.iter().map(|v| v.loss).sum::<f32>();

        let scale = 1.0 / batch_size as f32;

        for v in values.iter() {
            acc_gradient_by += &v.total_gradient_by;
            acc_gradient_bh += &v.total_gradient_bh;
            acc_gradient_W_xh += &v.total_gradient_W_xh;
            acc_gradient_W_hh += &v.total_gradient_W_hh;
            acc_gradient_W_hy += &v.total_gradient_W_hy;
        }
        //self.acc_weights(
        //    &mut acc_gradient_W_hh,
        //    values
        //        .iter()
        //        .map(|v| v.total_gradient_W_hh.view())
        //        .collect(),
        //);

        acc_gradient_W_hh *= scale;
        acc_gradient_W_xh *= scale;
        acc_gradient_W_hy *= scale;
        acc_gradient_bh *= scale;
        acc_gradient_by *= scale;

        self.W_xh -= &(lr * acc_gradient_W_xh);
        self.W_hh -= &(lr * acc_gradient_W_hh);
        self.W_hy -= &(lr * acc_gradient_W_hy);

        self.bh -= &(lr * acc_gradient_bh);
        self.by -= &(lr * acc_gradient_by);
        //}

        acc_loss / batch_size as f32
    }

    fn acc_weights<D: Dimension>(&self, acc: &mut Array<f32, D>, weights: Vec<ArrayView<f32, D>>) {
        for w in weights {
            *acc += &w;
        }
    }

    pub fn clear_hidden(&mut self) {
        self.h = Array1::zeros(self.h.dim());
    }

    pub fn load_weights(&mut self, path: &str) -> Result<()> {
        let file = fs::File::open(path)?;

        let data: FilePayload = serde_json::from_reader(file).unwrap();

        self.W_emb = vec_2d_to_arr(self.W_emb.shape(), data.W_emb);
        self.W_xh = vec_2d_to_arr(self.W_xh.shape(), data.W_xh);
        self.W_hh = vec_2d_to_arr(self.W_hh.shape(), data.W_hh);
        self.W_hy = vec_2d_to_arr(self.W_hy.shape(), data.W_hy);

        Ok(())
    }
    pub fn save_weights(&self, path: &str) -> Result<()> {
        let file = fs::File::create(path)?;

        let w_hh: Vec<Vec<f32>> = self.W_hh.outer_iter().map(|r| r.to_vec()).collect();
        let w_xh: Vec<Vec<f32>> = self.W_xh.outer_iter().map(|r| r.to_vec()).collect();
        let w_hy: Vec<Vec<f32>> = self.W_hy.outer_iter().map(|r| r.to_vec()).collect();
        let w_emb: Vec<Vec<f32>> = self.W_emb.outer_iter().map(|r| r.to_vec()).collect();

        let data = FilePayload {
            W_hh: w_hh,
            W_xh: w_xh,
            W_hy: w_hy,
            W_emb: w_emb,
        };

        let writer = BufWriter::new(file);

        serde_json::to_writer(writer, &data)?;

        Ok(())
    }
}

fn vec_2d_to_arr(shape: &[usize], vec: Vec<Vec<f32>>) -> Array2<f32> {
    let mut arr = Array2::zeros((0, shape[1]));

    vec.into_iter()
        .for_each(|r| arr.push_row(Array1::from_vec(r).view()).unwrap());

    assert_eq!(arr.shape(), shape);
    arr
}

#[derive(Serialize, Deserialize)]
struct FilePayload {
    #[allow(non_snake_case)]
    W_hh: Vec<Vec<f32>>,
    #[allow(non_snake_case)]
    W_xh: Vec<Vec<f32>>,
    #[allow(non_snake_case)]
    W_hy: Vec<Vec<f32>>,
    #[allow(non_snake_case)]
    W_emb: Vec<Vec<f32>>,
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

fn sample(probs: Array1<f32>) -> usize {
    let dist = WeightedIndex::new(probs).unwrap();

    let mut rng = rand::rng();

    dist.sample(&mut rng)
}

pub fn argmax(arr: &Array1<f32>) -> usize {
    arr.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}
