use std::{collections::HashMap, fs, mem::replace};

use clap::Parser;
use colored::Colorize;
use ndarray::Array1;
use serde::{Deserialize, Serialize, ser::SerializeStruct};
use ssm::{
    embedding2::EmbeddingChar,
    rnn::{ForwardOutput, RNN},
    utils::{corpus_file, corpus_folder, corpus_json, read_csv},
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

#[derive(Debug, Deserialize)]
pub struct QuoraJson {
    pub question: String,
    pub answer: String,
}

fn main() {
    let hidden_size = 60;

    //let corpus = corpus_folder("./corpus/dahl");
    let corpus = corpus_folder("./corpus/seuss");
    //let corpus = corpus1 + &corpus2;

    //let corpus = corpus_file("./corpus/seuss/the-cat-in-a-hat.txt");
    //let corpus = corpus_file("./corpus/dahl/charlie.txt");

    //let qa: QuoraJson = corpus_json("./corpus/quora.json").unwrap();

    //let records: Vec<DailyDialogRecord> = read_csv("./corpus/daily_dialog/train.csv").unwrap();
    //let corpus = records
    //    .into_iter()
    //    .take(50)
    //    .map(|v| {
    //        v.dialog
    //            .replace("['", "")
    //            .replace("']", "")
    //            .replace("[\"", "")
    //            .replace("\"]", "")
    //            .replace("\" ", "")
    //            .split(" '")
    //            .map(|v| v.to_string())
    //            .collect::<Vec<String>>()
    //            .join(" ")
    //    })
    //    .collect::<Vec<String>>()
    //    .join("\n");

    println!("{:?}", corpus);
    const DIM: usize = 300;
    const EPOCHS: usize = 600;

    println!("Corpus len {}", corpus.len());

    // STACKED, LTSM, GRU, CUDA, GPU,...
    let mut rnn = RNN::new(hidden_size, &corpus, DIM, 1);

    let weight_file = "./weights/seuss-60-10-1-popo";

    let args = Args::parse();
    if args.train {
        rnn.train_corpus(&corpus, EPOCHS, 1, 0.002, 20);
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
    let seed = String::from("Where did you sit?");
    let res = rnn.inference(&seed, 200, 0.9);

    println!();
    println!("'{}{}'", seed.yellow(), res.green());

    //let ForwardOutput { pred_y: y, .. } = rnn.forward(Array1::zeros(rnn.vocab_size).view());
    //let ForwardOutput { pred_y: y2, .. } = rnn.forward(y.view());
    //let ForwardOutput { pred_y: y3, .. } = rnn.forward(y2.view());
}
