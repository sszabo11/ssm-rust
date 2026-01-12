use std::{collections::HashMap, fs};

use clap::Parser;
use colored::Colorize;
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use ssm::{
    embedding2::EmbeddingChar,
    rnn::{ForwardOutput, RNN},
    utils::{corpus_file, corpus_folder, read_csv},
};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value_t = false)]
    train: bool,

    #[arg(short, long, default_value = "")]
    corpus: String,
}

#[derive(Serialize, Deserialize)]
struct DailyDialogRecord {
    dialog: String,
}

fn main() {
    let hidden_size = 512;

    let corpus = corpus_folder("./corpus/seuss");

    //let corpus = corpus_file("./corpus/seuss/the-cat-in-a-hat.txt");

    //let records: Vec<DailyDialogRecord> = read_csv("./corpus/daily_dialog/train.csv").unwrap();

    //let corpus = records
    //    .into_iter()
    //    .map(|v| v.dialog)
    //    .collect::<Vec<String>>()
    //    .join("\n");

    const DIM: usize = 400;
    const EPOCHS: usize = 500;

    // STACKED, LTSM, GRU, CUDA, GPU,...
    let mut rnn = RNN::new(hidden_size, &corpus, DIM);

    let weight_file = "./weights/seuss-3000-001-32-100";

    let args = Args::parse();
    if args.train {
        rnn.train_corpus(&corpus, EPOCHS, 16, 0.01, 100);
    } else {
        rnn.load_weights(weight_file)
            .expect("Failed to load weights");
    }

    println!("TRAINED");

    if args.train {
        rnn.save_weights(weight_file)
            .expect("Failed to save weights");
    }

    println!("vocab size: {}", rnn.vocab_size);

    rnn.clear_hidden();
    let seed = String::from("Horton");
    let res = rnn.inference(&seed, 200, 0.9);

    println!();
    println!("'{}{}'", seed.yellow(), res.green());

    //let ForwardOutput { pred_y: y, .. } = rnn.forward(Array1::zeros(rnn.vocab_size).view());
    //let ForwardOutput { pred_y: y2, .. } = rnn.forward(y.view());
    //let ForwardOutput { pred_y: y3, .. } = rnn.forward(y2.view());
}
