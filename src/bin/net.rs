use std::{collections::HashMap, fs};

use clap::Parser;
use colored::Colorize;
use ndarray::Array1;
use ssm::{
    embedding2::EmbeddingChar,
    rnn::{ForwardOutput, RNN},
    utils::{corpus_file, corpus_folder},
};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value_t = false)]
    train: bool,

    #[arg(short, long, default_value = "")]
    corpus: String,
}

fn main() {
    let hidden_size = 100;

    let corpus = corpus_folder("./corpus/seuss");

    //let corpus = corpus_file("./corpus/seuss/the-cat-in-a-hat.txt");

    const DIM: usize = 300;
    const EPOCHS: usize = 300;

    let mut rnn = RNN::new(hidden_size, &corpus, DIM);

    let args = Args::parse();
    if args.train {
        for epoch in 0..EPOCHS {
            println!("Epoch: {}", epoch);
            for sentence in corpus.split_terminator(".") {
                let tokens: Vec<char> = sentence.chars().collect();

                let idxs: Vec<usize> = tokens
                    .iter()
                    .map(|c| rnn.char_to_i.get(c).copied().expect("Char not found"))
                    .collect();

                rnn.train(Array1::from_vec(idxs), 10, 0.003);
            }
        }
    } else {
        rnn.load_weights("./weights/1")
            .expect("Failed to load weights");
    }

    println!("TRAINED");

    // rnn.save_weights("./weights/1")
    //     .expect("Failed to save weights");

    println!("vocab size: {}", rnn.vocab_size);

    rnn.clear_hidden();
    let seed = String::from("Hello");
    let res = rnn.inference(&seed, 400);

    println!();
    println!("'{}{}'", seed.yellow(), res.green());

    //let ForwardOutput { pred_y: y, .. } = rnn.forward(Array1::zeros(rnn.vocab_size).view());
    //let ForwardOutput { pred_y: y2, .. } = rnn.forward(y.view());
    //let ForwardOutput { pred_y: y3, .. } = rnn.forward(y2.view());
}
