use std::{collections::HashMap, fs};

use ndarray::Array1;
use ssm::{
    embedding2::EmbeddingChar,
    rnn::{ForwardOutput, RNN},
    utils::{corpus_file, corpus_folder},
};

fn main() {
    let hidden_size = 100;
    let seq_length = 25;

    //let corpus = corpus_folder("./corpus/seuss");

    let corpus = corpus_file("./corpus/seuss/the-cat-in-a-hat.txt");

    const DIM: usize = 300;
    const SL_WIN: usize = 5;
    const K: usize = 10;
    const EPOCHS: usize = 200;

    //let embedding = EmbeddingChar::new(corpus.clone(), DIM, SL_WIN, K);

    let mut rnn = RNN::new(hidden_size, seq_length, &corpus, DIM, SL_WIN, K);

    for epoch in 0..EPOCHS {
        println!("Epoch: {}", epoch);
        for sentence in corpus.split_terminator(".") {
            let tokens: Vec<char> = sentence.chars().collect();

            let idxs: Vec<usize> = tokens
                .iter()
                .map(|c| rnn.char_to_i.get(c).copied().expect("Char not found"))
                .collect();

            println!("{:?}", idxs);

            rnn.train(Array1::from_vec(idxs), 10, 0.01);
        }
    }

    println!("TRAINED");
    println!("vocab size: {}", rnn.vocab_size);

    rnn.clear_hidden();
    let res = rnn.inference("The cat i", 100);

    println!();
    println!("'{}'", res);

    //let ForwardOutput { pred_y: y, .. } = rnn.forward(Array1::zeros(rnn.vocab_size).view());
    //let ForwardOutput { pred_y: y2, .. } = rnn.forward(y.view());
    //let ForwardOutput { pred_y: y3, .. } = rnn.forward(y2.view());
}
