use std::collections::HashMap;

use ndarray::{Array, Array1, Array2, ArrayView, ArrayView1, Ix1, Ix2, ShapeArg};
use ndarray_rand::RandomExt;
use rand::distr::Uniform;

pub struct RNN {
    #[allow(non_snake_case)]
    W_hh: Array2<f32>, // Weight for hidden - hidden
    #[allow(non_snake_case)]
    W_xh: Array2<f32>, // Weight for input - hidden
    #[allow(non_snake_case)]
    W_hy: Array2<f32>, // Weight for hidden - output

    by: Array1<f32>,
    bh: Array1<f32>,

    h: Array1<f32>,
    hidden_size: usize,
    seq_length: usize,
    pub vocab_size: usize,
}

impl RNN {
    pub fn new(hidden_size: usize, seq_length: usize, corpus: &str) -> Self {
        let vocab_size = corpus.len();

        Self {
            vocab_size,
            hidden_size,
            seq_length,
            h: Array1::zeros(hidden_size),
            W_hy: Array2::random((vocab_size, hidden_size), Uniform::new(-0.1, 0.1).unwrap()),
            W_xh: Array2::random((hidden_size, vocab_size), Uniform::new(-0.1, 0.1).unwrap()),
            W_hh: Array2::random((hidden_size, hidden_size), Uniform::new(-0.1, 0.1).unwrap()),
            bh: Array1::zeros(hidden_size),
            by: Array1::zeros(vocab_size),
        }
    }

    pub fn forward(&mut self, x: ArrayView1<f32>) -> (Array1<f32>, Array1<f32>, Array1<f32>) {
        // Pre-Activation (logits or linear output)
        let z_t = self.W_hh.dot(&self.h) + self.W_xh.dot(&x) + &self.bh;

        // Hidden state
        let h_t = z_t.tanh();

        self.h = h_t.clone();
        // Prediction
        let y = self.W_hy.dot(&self.h) + &self.by;

        (y, z_t, h_t)
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

    pub fn train(&mut self, input: Array2<f32>, target: Array2<f32>, batch_size: usize, lr: usize) {
        for (i, x) in input.rows().into_iter().enumerate() {
            let target_y = target.row(i);

            // Cache the predictions for backprop
            let mut pred_y_vec = Vec::new();
            let mut pred_z_vec = Vec::new();
            let mut total_gradient_W_xh = Array1::<f32>::zeros(self.hidden_size);
            let mut total_gradient_W_hh = Array1::<f32>::zeros(self.hidden_size);
            let mut total_gradient_W_hy = Array1::<f32>::zeros(self.vocab_size);

            let mut total_gradient_by = Array1::zeros(self.vocab_size);
            let mut total_gradient_bh = Array1::zeros(self.hidden_size);

            // Forward
            for t in 0..self.seq_length {
                self.clear_hidden();
                let (pred_y, z_t, h_t) = self.forward(x.t());
                pred_y_vec.push(pred_y);
                pred_z_vec.push(z_t);
            }

            // Backprop
            for t in (0..self.seq_length).rev() {
                let pred_y = &pred_y_vec[t];
                let z = &pred_z_vec[t];

                let delta_y_t = pred_y - &target_y;
                // Propagate error to hidden state
                let w_hy_T = self.W_hy.to_shape((self.W_hy.nrows(), 1)).unwrap();
                //let delta_h_t = w_hy_T.dot(&delta_y_t) + delta_h_t1;

                let loss = self.loss(target_y.t(), pred_y.t(), batch_size);

                //total_gradient_W_xh += &(&delta_y * self.h[t]);

                //total_gradient_W_hh += &(z * &x);
                total_gradient_W_hy += &(outer(delta_y_t, self.h));
            }
            self.W_xh -= &(lr as f32 * total_gradient_W_xh);
            self.W_hh -= &(lr as f32 * total_gradient_W_hh);
            self.W_hy -= &(lr as f32 * total_gradient_W_hy);

            self.bh -= &(lr as f32 * total_gradient_bh);
            self.by -= &(lr as f32 * total_gradient_by);
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

pub fn outer(x: &Array<f64, Ix1>, y: &Array<f64, Ix1>) -> Array<f64, Ix1> {
    let (size_x, size_y) = (x.shape()[0], y.shape()[0]);
    let x_view = x.view();
    let y_view = y.view();
    let x_reshaped = x_view.to_shape((size_x, 1)).unwrap();
    let y_reshaped = y_view.to_shape((size_y, 1)).unwrap();
    x_reshaped.dot(&y_reshaped)
}
