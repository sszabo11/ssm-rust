#![allow(non_snake_case)]

use std::{
    collections::HashMap,
    f64,
    fs::{self},
    io::BufWriter,
};

use anyhow::Result;
use ndarray::{Array, Array1, Array2, Array3, ArrayView2, Axis, Dimension, Ix1, Ix2};
use ndarray_rand::RandomExt;
use rand::distr::{Distribution, Uniform, weighted::WeightedIndex};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

use crate::embedding2::get_chars;

#[allow(clippy::upper_case_acronyms)]
pub struct RNN {
    W_hh: Vec<Array2<f32>>, // Weight for hidden - hidden
    W_xh: Vec<Array2<f32>>, // Weight for input - hidden
    W_hy: Array2<f32>,      // Weight for hidden - output
    W_emb: Array2<f32>,     // Weight for embedding

    by: Array1<f32>,
    bh: Vec<Array1<f32>>,

    h: Vec<Array1<f32>>,

    hidden_size: usize,
    embedding_dim: usize,
    pub layers: usize,
    pub vocab_size: usize,
    pub char_to_i: HashMap<char, usize>,
    pub i_to_char: HashMap<usize, char>,
}

#[derive(Debug)]
pub struct ForwardOutput {
    pub h_t: Vec<Array1<f32>>,
    pub z_layers: Vec<Array1<f32>>,
    pub x_t: Array1<f32>,
    pub target_idx: usize,
    pub logits_t: Array1<f32>,
}

pub struct MiniBatch {
    total_gradient_W_xh: Vec<Array2<f32>>,
    total_gradient_W_hh: Vec<Array2<f32>>,
    total_gradient_W_hy: Array2<f32>,
    total_gradient_by: Array1<f32>,
    total_gradient_bh: Vec<Array1<f32>>,
    loss: f32,
}

impl RNN {
    pub fn new(hidden_size: usize, corpus: &str, embedding_dim: usize, layers: usize) -> Self {
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

        let mut xh = Vec::new();

        xh.push(Array2::random(
            (hidden_size, embedding_dim), // First input is embedding
            Uniform::new(-0.1, 0.1).unwrap(),
        ));

        for _ in 1..layers {
            xh.push(Array2::random(
                (hidden_size, hidden_size), // Internal layers have hidden state passed so hidden size
                Uniform::new(-0.1, 0.1).unwrap(),
            ));
        }

        Self {
            vocab_size,
            hidden_size,
            char_to_i,
            i_to_char,
            embedding_dim,
            layers,
            h: (0..layers).map(|_| Array1::zeros(hidden_size)).collect(),
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
            W_xh: xh,
            W_hh: (0..layers)
                .map(|_| {
                    Array2::random((hidden_size, hidden_size), Uniform::new(-0.1, 0.1).unwrap())
                })
                .collect(),
            bh: (0..layers).map(|_| Array1::zeros(hidden_size)).collect(),
            by: Array1::zeros(vocab_size),
        }
    }

    pub fn forward(&self, input: &Array1<usize>) -> Vec<ForwardOutput> {
        let seq_len = input.len();
        let mut output: Vec<ForwardOutput> = Vec::with_capacity(input.len());

        let mut prev_h = vec![Array1::zeros(self.hidden_size); self.layers];

        for t in 0..(seq_len - 1) {
            let curr_idx = input[t];
            let next_idx = input[t + 1];

            let mut curr_input = self.W_emb.row(curr_idx).to_owned();
            let mut current_h_layers = vec![Array1::zeros(self.hidden_size); self.layers];
            let mut current_z_layers = vec![Array1::zeros(self.hidden_size); self.layers];
            //let mut z_layers = Vec::with_capacity(self.layers);

            for l in 0..self.layers {
                // Pre-Activation (logits or linear output)
                let z_t =
                    self.W_hh[l].dot(&prev_h[l]) + self.W_xh[l].dot(&curr_input) + &self.bh[l];
                current_z_layers[l] = z_t.clone();
                //z_layers.push(z_t.clone());
                let h_t = z_t.mapv(sigmoid);
                current_h_layers[l] = h_t.clone();

                curr_input = current_h_layers[l].clone();
            }

            // Output logits
            // Top (last layer) hidden state
            let logits_t = self.W_hy.dot(&current_h_layers[self.layers - 1]) + &self.by;

            output.push(ForwardOutput {
                h_t: current_h_layers.clone(),
                z_layers: current_z_layers,
                x_t: self.W_emb.row(curr_idx).to_owned(),
                logits_t,
                target_idx: next_idx,
            });
            prev_h = current_h_layers
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
            let mut curr_x = self.W_emb.row(*char_idx).to_owned();
            //println!("{:?} {:?} {}", self.W_xh[0].dim(), self.W_xh[1].dim(), curr_x.dim());
            for l in 0..self.layers {
                let z_t = self.W_hh[l].dot(&self.h[l]) + self.W_xh[l].dot(&curr_x) + &self.bh[l];

                let a_t = z_t.mapv(sigmoid);

                self.h[l] = a_t.clone();
                curr_x = a_t
            }
        }

        while response.len() < len {
            let logits = self.W_hy.dot(&self.h[self.layers - 1]) + &self.by;

            let probs = softmax(&logits);

            //println!("probs: {}", probs);
            let temp_probs: Array1<f32> = probs.iter().map(|v| v / temp).collect();

            let next_idx = sample(temp_probs);

            let char = self.i_to_char.get(&next_idx).unwrap();
            response.push(*char);

            let mut curr_x = self.W_emb.row(next_idx).to_owned();
            //println!("{:?} {:?} {}", self.W_xh[0].dim(), self.W_xh[1].dim(), curr_x.dim());
            for l in 0..self.layers {
                let z_t = self.W_hh[l].dot(&self.h[l]) + self.W_xh[l].dot(&curr_x) + &self.bh[l];

                let a_t = z_t.mapv(sigmoid);

                self.h[l] = a_t.clone();
                curr_x = a_t;
            }
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

        for epoch in 1..=epochs {
            let mut loss = 0.0;

            for batch in batch_input.outer_iter() {
                let batch_loss = self.train_mini_batch(&batch, lr);
                loss = batch_loss;
            }

            //if epoch % 100 == 0 {
            println!("Epoch: {} | Loss: {}", epoch, loss);
            //}
        }
    }

    pub fn train_mini_batch(
        &mut self,
        inputs: &ArrayView2<usize>, // Single batch
        lr: f32,
    ) -> f32 {
        let batch_size = inputs.nrows();
        let mut acc_loss = 0.0;

        let mut acc_gradient_W_xh = Vec::new();
        acc_gradient_W_xh.push(Array2::<f32>::zeros((self.hidden_size, self.embedding_dim)));
        for _ in 1..self.layers {
            acc_gradient_W_xh.push(Array2::<f32>::zeros((self.hidden_size, self.hidden_size)));
        }

        let mut acc_gradient_W_hh =
            vec![Array2::<f32>::zeros((self.hidden_size, self.hidden_size)); self.layers];
        let mut acc_gradient_W_hy = Array2::<f32>::zeros((self.vocab_size, self.hidden_size));
        let mut acc_gradient_by = Array1::zeros(self.vocab_size);
        let mut acc_gradient_bh = vec![Array1::zeros(self.hidden_size); self.layers];

        // Each sequence in batch
        let values: Vec<MiniBatch> = (0..batch_size)
            .collect::<Vec<usize>>()
            .into_par_iter()
            .map(|batch| {
                let input = inputs.row(batch).to_owned();

                let mut total_gradient_W_xh = Vec::new();

                total_gradient_W_xh
                    .push(Array2::<f32>::zeros((self.hidden_size, self.embedding_dim)));
                for _ in 1..self.layers {
                    total_gradient_W_xh
                        .push(Array2::<f32>::zeros((self.hidden_size, self.hidden_size)));
                }

                let mut total_gradient_W_hh =
                    vec![Array2::<f32>::zeros((self.hidden_size, self.hidden_size)); self.layers];

                let mut total_gradient_W_hy =
                    Array2::<f32>::zeros((self.vocab_size, self.hidden_size));

                let mut total_gradient_by = Array1::zeros(self.vocab_size);
                let mut total_gradient_bh = vec![Array1::zeros(self.hidden_size); self.layers];

                if input.is_empty() {
                    println!("Empty");
                };

                let forward_outputs = self.forward(&input);
                let loss = self.loss(&forward_outputs, input);

                let mut delta_from_above = Array1::zeros(self.vocab_size); // start from output
                for l in (0..self.layers).rev() {
                    let mut delta_h_next = Array1::zeros(self.hidden_size);

                    for t in (0..forward_outputs.len()).rev() {
                        //let mut delta_z_from_above = Array1::zeros(self.hidden_size);
                        let target_idx = &forward_outputs[t].target_idx;

                        let h_t = &forward_outputs[t].h_t[l]; // Hidden state

                        let h_prev = if t == 0 {
                            h_t
                        } else {
                            &forward_outputs[t - 1].h_t[l]
                        };

                        let z_t = &forward_outputs[t].z_layers[l]; // Pre-Activation
                        let sig_der = z_t.map(|z| sigmoid_der(*z)); // Sigmoid derivative element wise

                        let x_t = if l == 0 {
                            &forward_outputs[t].x_t // Input state
                        } else {
                            &forward_outputs[t].h_t[l - 1] // Input state
                        };

                        if l == self.layers - 1 {
                            let logits_t = &forward_outputs[t].logits_t;

                            let probs = softmax(logits_t);
                            let mut delta_y = probs;
                            *delta_y.get_mut(*target_idx).unwrap() -= 1.0;
                            delta_from_above = delta_y.clone();

                            let gradient_hy = outer_product(&delta_y, h_t);

                            assert_eq!(gradient_hy.shape(), &[self.vocab_size, self.hidden_size]);

                            total_gradient_W_hy += &gradient_hy;
                            total_gradient_by += &delta_y;
                        }
                        let delta_h = if l == self.layers - 1 {
                            self.W_hy.t().dot(&delta_from_above) + &delta_h_next
                        } else {
                            self.W_xh[l + 1].t().dot(&delta_from_above) + &delta_h_next
                        };

                        let delta_z = &delta_h * sig_der; // derivative dL/da(L) or dL/dy

                        {
                            // THIS uses x_t
                            // For w_xh
                            let gradient_xh = outer_product(&delta_z, x_t);

                            if l == 0 {
                                assert_eq!(
                                    gradient_xh.shape(),
                                    &[self.hidden_size, self.embedding_dim]
                                );
                            } else {
                                assert_eq!(
                                    gradient_xh.shape(),
                                    &[self.hidden_size, self.hidden_size]
                                );
                            }
                            total_gradient_W_xh[l] += &gradient_xh;
                        }

                        {
                            // THIS USES h{t-1}
                            // For w_hh
                            let gradient_hh = outer_product(&delta_z, h_prev);

                            assert_eq!(gradient_hh.shape(), &[self.hidden_size, self.hidden_size]);

                            total_gradient_W_hh[l] += &gradient_hh;
                        }
                        {
                            // THIS USES h_t
                            // For w_hy
                        }

                        {
                            // THIS USES
                            // For bh

                            total_gradient_bh[l] += &delta_z;
                        }

                        {
                            // THIS USES
                            // For by
                        }

                        delta_h_next = self.W_hh[l].t().dot(&delta_z);
                        delta_from_above = delta_h; // or delta_z
                        //if l != self.layers - 1 {
                        //    delta_z_from_above = self.W_xh[l + 1].t().dot(&delta_z);
                        //}
                    }
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
            acc_gradient_W_hy += &v.total_gradient_W_hy;
            acc_gradient_by += &v.total_gradient_by;
            for l in 0..self.layers {
                acc_gradient_bh[l] += &v.total_gradient_bh[l];
                acc_gradient_W_xh[l] += &v.total_gradient_W_xh[l];
                acc_gradient_W_hh[l] += &v.total_gradient_W_hh[l];
            }
        }

        acc_gradient_W_hy *= scale;
        acc_gradient_by *= scale;

        for l in 0..self.layers {
            acc_gradient_W_hh[l] *= scale;
            acc_gradient_W_xh[l] *= scale;
            acc_gradient_bh[l] *= scale;
        }

        for l in 0..self.layers {
            self.W_xh[l] -= &(lr * &acc_gradient_W_xh[l]);
            self.W_hh[l] -= &(lr * &acc_gradient_W_hh[l]);
            self.bh[l] -= &(lr * &acc_gradient_bh[l]);
        }
        self.W_hy -= &(lr * &acc_gradient_W_hy);
        self.by -= &(lr * &acc_gradient_by);

        acc_loss / batch_size as f32
    }

    //fn acc_weights<D: Dimension>(&self, acc: &mut Array<f32, D>, weights: Vec<ArrayView<f32, D>>) {
    //    for w in weights {
    //        *acc += &w;
    //    }
    //}

    pub fn clear_hidden(&mut self) {
        self.h = vec![Array1::zeros(self.h[0].dim()); self.layers];
    }

    pub fn load_weights(&mut self, path: &str) -> Result<()> {
        let file = fs::File::open(path)?;

        let data: FilePayload = serde_json::from_reader(file).unwrap();

        self.W_emb = vec_2d_to_arr(&[data.vocab_size, data.embedding_dim], data.W_emb);
        self.W_xh = vec_3d_to_arr(&[data.hidden_size, data.embedding_dim], data.W_xh);
        self.W_hh = vec_3d_to_arr(&[data.hidden_size, data.hidden_size], data.W_hh);
        self.W_hy = vec_2d_to_arr(&[data.vocab_size, data.hidden_size], data.W_hy);

        self.bh = data.bh.into_iter().map(Array1::from_vec).collect();
        self.by = Array1::from_vec(data.by);

        Ok(())
    }
    pub fn save_weights(&self, path: &str) -> Result<()> {
        let file = fs::File::create(path)?;

        let w_hh: Vec<Vec<Vec<f32>>> = self
            .W_hh
            .iter()
            .map(|w| w.outer_iter().map(|r| r.to_vec()).collect())
            .collect();
        let w_xh: Vec<Vec<Vec<f32>>> = self
            .W_xh
            .iter()
            .map(|w| w.outer_iter().map(|r| r.to_vec()).collect())
            .collect();

        let w_hy: Vec<Vec<f32>> = self.W_hy.outer_iter().map(|r| r.to_vec()).collect();

        let w_emb: Vec<Vec<f32>> = self.W_emb.outer_iter().map(|r| r.to_vec()).collect();

        let data = FilePayload {
            vocab_size: self.vocab_size,
            embedding_dim: self.embedding_dim,
            hidden_size: self.hidden_size,
            W_hh: w_hh,
            W_xh: w_xh,
            W_hy: w_hy,
            W_emb: w_emb,
            by: self.by.to_vec(),
            bh: self.bh.iter().map(|v| v.to_vec()).collect(),
        };

        let writer = BufWriter::new(file);

        serde_json::to_writer(writer, &data)?;

        Ok(())
    }
}

fn vec_2d_to_arr(shape: &[usize], vec: Vec<Vec<f32>>) -> Array2<f32> {
    let mut arr = Array2::zeros((0, shape[1]));

    vec.into_iter().for_each(|r| {
        arr.push_row(Array1::from_vec(r).view()).unwrap();
    });

    assert_eq!(arr.shape(), shape);
    arr
}

fn vec_3d_to_arr(shape: &[usize], vec: Vec<Vec<Vec<f32>>>) -> Vec<Array2<f32>> {
    let mut v = Vec::new();
    //let mut arr = Array2::zeros((0, shape[1]));

    vec.into_iter().for_each(|l| {
        let mut arr = Array2::zeros((0, shape[1]));
        l.into_iter()
            .for_each(|r| arr.push_row(Array1::from_vec(r).view()).unwrap());

        v.push(arr);
    });

    assert_eq!(v[0].shape(), shape);
    v
}

#[derive(Serialize, Deserialize)]
struct FilePayload {
    #[allow(non_snake_case)]
    W_hh: Vec<Vec<Vec<f32>>>,
    #[allow(non_snake_case)]
    W_xh: Vec<Vec<Vec<f32>>>,
    #[allow(non_snake_case)]
    W_hy: Vec<Vec<f32>>,
    #[allow(non_snake_case)]
    W_emb: Vec<Vec<f32>>,

    embedding_dim: usize,
    hidden_size: usize,
    vocab_size: usize,
    by: Vec<f32>,
    bh: Vec<Vec<f32>>,
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
