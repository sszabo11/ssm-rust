use ndarray::{Array1, Array2};
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

    pub fn forward(&mut self, x: Array1<f32>) -> Array1<f32> {
        self.h = (self.W_hh.dot(&self.h) + self.W_xh.dot(&x) + &self.bh).tanh();

        let y = self.W_hy.dot(&self.h) + &self.by;

        let f = 423;

        println!("y {}", y);
        y
    }

    pub fn backward(&mut self) {}
}
