use std::{collections::HashMap, fs};

use ndarray::Array1;
use ssm::{
    embedding2::EmbeddingChar,
    rnn::{ForwardOutput, RNN},
    utils::corpus_folder,
};

fn main() {
    let hidden_size = 100;
    let seq_length = 25;

    let corpus = corpus_folder("./corpus/seuss");

    const DIM: usize = 300;
    const SL_WIN: usize = 5;
    const K: usize = 10;

    //let embedding = EmbeddingChar::new(corpus.clone(), DIM, SL_WIN, K);

    let mut rnn = RNN::new(hidden_size, seq_length, &corpus, DIM, SL_WIN, K);

    let mut char_to_i: HashMap<char, usize> = HashMap::new();
    let mut i_to_char: HashMap<usize, char> = HashMap::new();

    let mut unique_chars: Vec<char> = corpus.chars().collect();
    unique_chars.sort_unstable();
    unique_chars.dedup();

    for (idx, c) in unique_chars.into_iter().enumerate() {
        char_to_i.insert(c, idx);
        i_to_char.insert(idx, c);
    }

    for sentence in corpus.split_terminator(".") {
        let tokens: Vec<char> = sentence.chars().collect();

        let idxs: Vec<usize> = tokens
            .iter()
            .map(|c| *char_to_i.get(c).expect("Char not found"))
            .collect();

        rnn.train(Array1::from_vec(idxs), 10, 0.0001);
    }

    //let ForwardOutput { pred_y: y, .. } = rnn.forward(Array1::zeros(rnn.vocab_size).view());
    //let ForwardOutput { pred_y: y2, .. } = rnn.forward(y.view());
    //let ForwardOutput { pred_y: y3, .. } = rnn.forward(y2.view());
}
