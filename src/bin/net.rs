use std::fs;

use ndarray::Array1;
use ssm::{rnn::RNN, utils::corpus_folder};

fn main() {
    let hidden_size = 100;
    let seq_length = 25;

    let corpus = corpus_folder("./corpus/seuss");

    let mut rnn = RNN::new(hidden_size, seq_length, &corpus);

    let y = rnn.step(Array1::zeros(rnn.vocab_size));
    let y2 = rnn.step(y);
    let y3 = rnn.step(y2);
}
