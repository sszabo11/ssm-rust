mod embedding;
mod embedding2;
mod ssm;
mod train;
mod utils;

use std::collections::HashMap;
use std::fs;

use crate::embedding2::EmbeddingChar;
use crate::train::{prepare_training_data, train, train_with_optimal};
use crate::{embedding::Embedding, ssm::SSM, train::generate_sequences, utils::corpus_folder};
use clap::Parser;
use clap::ValueEnum;
use ndarray::Array1;

#[derive(ValueEnum, Debug, Clone)] // ArgEnum here
#[clap(rename_all = "kebab_case")]
enum TrainParam {
    Embedding,
    SSM,
    Both,
    None,
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    train: TrainParam,

    #[arg(short, long, default_value = "")]
    corpus: String,

    #[arg(short, long, default_value_t = false)]
    find_optimal: bool,
}

fn main() {
    let args = Args::parse();

    let corpus = corpus_folder("./corpus/seuss");
    let mut embedding = EmbeddingChar::new(corpus, 500, 5, 10);

    match args.train {
        TrainParam::Embedding => {
            embedding.train(100, 0.000001, 0.001);

            embedding
                .save_embeddings_npy("./embeddings/chars-3.npy")
                .unwrap();
        }
        _ => {
            let matrix = embedding
                .load_embeddings_npy("./embeddings/chars-3.npy")
                .unwrap();
            embedding.set_input(matrix);
        }
    }

    let seed = "the cat i";
    let out1 = embedding.predict2(seed, 30, 2, 0.9);
    let out6 = embedding.predict2(seed, 30, 2, 0.1);
    let out2 = embedding.predict2(seed, 30, 2, 0.2);
    let out = embedding.predict(seed, 30, 8);
    let out7 = embedding.predict2(seed, 30, 2, 0.6);
    let out3 = embedding.predict2(seed, 30, 2, 0.4);
    let out4 = embedding.predict2(seed, 30, 2, 0.3);
    let out5 = embedding.predict2(seed, 30, 2, 0.5);

    println!("{}", out);
    println!("{}", out1);
    println!("{}", out2);
    println!("{}", out3);
    println!("{}", out4);
    println!("{}", out5);
    println!("{}", out6);
    println!("{}", out7);
    let encoded = embedding.encode_word("the cat");
    let decoded = embedding.decode_word(encoded, 0.1);

    println!("{}", decoded);

    //let out1 = embedding.predict2(seed, 3, 20, 0.7);
    //let out3 = embedding.predict2(seed, 3, 20, 0.1);
    //let out4 = embedding.predict2(seed, 3, 20, 0.4);
    //let out2 = embedding.predict(seed, 3, 20);

    //println!("{}", out1);
    //println!("{}", out2);
    //println!("{}", out3);
    //println!("{}", out4);

    return;

    // Load/Train embeddings

    println!("Corpus len: {}", corpus.len());
    println!(
        "Corpus words: {}",
        corpus.split_whitespace().collect::<Vec<&str>>().len()
    );

    let vocab_words = vec![
        "the", "a", "cat", "dog", "bird", "sat", "ran", "flew", "on", "mat", "fast", "likes",
        "fish", "bone", "seed", "is", "black", "brown",
    ];
    const DIM: usize = 500;
    let mut model = Embedding::new(corpus.clone(), DIM, 4, 10);

    model.save_vocab("./embeddings/vocab.txt").unwrap();
    //println!("{:?}", model.vocab);

    //let res = model.predict("The", 400, 4);
    //println!("{}", res);

    // Load/Train SSM
    let mut ssm = SSM::new(DIM, 64);

    //let mut tr = generate_sequences(100, 20, 3);
    //let mut tr6 = generate_sequences(100, 20, 6);
    //let tr10 = generate_sequences(100, 20, 10);

    //tr6.extend(tr10);
    //tr.extend(tr6);

    let training_sentences = prepare_training_data(&corpus);

    println!("{:?}", training_sentences.iter().take(3));

    //let encoded_data = model.encode_data(training_sentences);
    match args.train {
        TrainParam::None => {
            let (w_to_i, _i_to_w, matrix) =
                model.load_embeddings_txt("./embeddings/seuss.txt").unwrap();
            model.set_input(matrix);
        }
        TrainParam::SSM => {
            if args.find_optimal {
                //ssm.train_full(tr, lr);
                //train_with_optimal(&mut ssm, &tr);
            } else {
                //let ts = training_sentences
                //    .into_iter()
                //    .map(|v| v.iter().map(|s| s.to_string()).collect::<Vec<String>>())
                //    .collect::<Vec<(Vec<String>, Vec<String>)>>();

                ssm.train_full(&training_sentences, 0.0009, &mut model);
                //train(&mut ssm, &tr);
            }
        }
        TrainParam::Embedding => {
            model.train(5, 0.00001, 0.00001);
            model.save_embeddings_txt("./embeddings/seuss.txt").unwrap();
        }
        TrainParam::Both => {
            model.train(10, 0.0001, 0.00001);
            model.save_embeddings_txt("./embeddings/seuss.txt").unwrap();
            if args.find_optimal {
                //train_with_optimal(&mut ssm, &tr);
            } else {
                ssm.train_full(&training_sentences, 0.0009, &mut model);
                ssm.save_weights("./weights/weights.txt").unwrap();
            }
        }
    }

    let start = "the cat";
    let out1 = ssm.predict_sequence(&[model.encode_word(start)], 30);

    let words = out1
        .iter()
        .map(|word_vec| model.decode_word(Array1::from_vec(word_vec.clone()), 1.0))
        .collect::<Vec<String>>();

    println!();
    println!("Result: {} {}", start, words.join(" "));
    println!();

    //let out1 = ssm.predict_sequence(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 10);
    //println!();
    //println!("Expected: {:?}", &[4.0, 5.0, 6.0, 7.0]);
    //println!("Result: {:?}", out1);
    //println!();

    //let mut values = vec![0.0, 1.0, 2.0, 3.0];
    //for _ in 0..10 {
    //    let v = &values
    //        .clone()
    //        .into_iter()
    //        .rev()
    //        .take(4)
    //        .rev()
    //        .collect::<Vec<f32>>();
    //    println!("V: {:?}", v);
    //    let out = ssm.predict_sequence(v, 1);
    //    values.push(out[0]);
    //}
    //println!("Result: {:?}", values);
    //println!();

    //let out2 = ssm.predict_sequence(&[3.0, 5.0, 7.0, 9.0], 4);
    //println!();
    //println!("Expected: {:?}", &[11.0, 13.0, 15.0, 17.0]);
    //println!("Result: {:?}", out2);
    //println!();

    //let out3 = ssm.predict_sequence(&[10.0, 20.0, 30.0], 3);
    //println!();
    //println!("Expected: {:?}", &[40.0, 50.0, 60.0]);
    //println!("Result: {:?}", out3);
    //println!();

    //println!();
    //let out4 = ssm.predict_sequence(&[10.0, 30.0, 50.0], 6);
    //println!("Expected: {:?}", &[70.0, 90.0, 110.0]);
    //println!("Result: {:?}", out4);
    //println!();
}
