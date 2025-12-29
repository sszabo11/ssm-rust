mod embedding;
mod ssm;
mod train;
mod utils;

use std::collections::HashMap;

use crate::train::{train, train_with_optimal};
use crate::{embedding::Embedding, ssm::SSM, train::generate_sequences, utils::corpus_folder};
use clap::Parser;
use clap::ValueEnum;

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

    // Load/Train embeddings
    let corpus = corpus_folder("./corpus/seuss");

    println!("Corpus len: {}", corpus.len());
    println!(
        "Corpus words: {}",
        corpus.split_whitespace().collect::<Vec<&str>>().len()
    );

    let vocab_words = vec![
        "the", "a", "cat", "dog", "bird", "sat", "ran", "flew", "on", "mat", "fast", "likes",
        "fish", "bone", "seed", "is", "black", "brown",
    ];
    const DIM: usize = 800;
    let mut model = Embedding::new(vocab_words.join(" "), DIM, 5, 10);
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

    //let training_sentences = vec![
    //    // Simple subject-verb-object patterns
    //    vec!["the", "cat", "sat"],
    //    vec!["the", "dog", "ran"],
    //    vec!["the", "bird", "flew"],
    //    vec!["the", "fish", "swam"],
    //    // With locations
    //    vec!["the", "cat", "sat", "on", "mat"],
    //    vec!["the", "dog", "ran", "in", "park"],
    //    vec!["the", "bird", "flew", "over", "house"],
    //    vec!["the", "fish", "swam", "in", "water"],
    //    // Variations
    //    vec!["cat", "sat", "on", "mat"],
    //    vec!["dog", "ran", "in", "park"],
    //    vec!["bird", "flew", "over", "house"],
    //    // More patterns
    //    vec!["the", "cat", "likes", "food"],
    //    vec!["the", "dog", "likes", "bone"],
    //    vec!["the", "bird", "likes", "seed"],
    //    // Longer sequences
    //    vec!["the", "cat", "sat", "on", "the", "mat"],
    //    vec!["the", "dog", "ran", "to", "the", "park"],
    //    vec!["the", "bird", "flew", "to", "the", "tree"],
    //    // Different starters
    //    vec!["a", "cat", "sat"],
    //    vec!["a", "dog", "ran"],
    //    vec!["a", "bird", "flew"],
    //    // Actions
    //    vec!["cat", "eats", "food"],
    //    vec!["dog", "eats", "bone"],
    //    vec!["bird", "eats", "seed"],
    //    // Colors
    //    vec!["the", "black", "cat", "sat"],
    //    vec!["the", "brown", "dog", "ran"],
    //    vec!["the", "blue", "bird", "flew"],
    //    // Simple descriptions
    //    vec!["cat", "is", "black"],
    //    vec!["dog", "is", "brown"],
    //    vec!["bird", "is", "blue"],
    //    vec!["sky", "is", "blue"],
    //    vec!["grass", "is", "green"],
    //    // More context
    //    vec!["the", "cat", "sleeps", "on", "bed"],
    //    vec!["the", "dog", "plays", "in", "yard"],
    //    vec!["the", "bird", "sings", "in", "tree"],
    //    // Repetition helps learning
    //    vec!["the", "cat", "sat"],
    //    vec!["the", "cat", "sat", "on", "mat"],
    //    vec!["cat", "sat", "on", "mat"],
    //    vec!["the", "dog", "ran"],
    //    vec!["the", "dog", "ran", "fast"],
    //    vec!["dog", "ran", "fast"],
    //    // Questions (for variety)
    //    vec!["where", "is", "cat"],
    //    vec!["where", "is", "dog"],
    //    // Possessives
    //    vec!["my", "cat", "sat"],
    //    vec!["my", "dog", "ran"],
    //    vec!["his", "bird", "flew"],
    //];

    let training_sentences = vec![
        vec!["the", "cat", "sat"],
        vec!["the", "cat", "sat", "on", "mat"],
        vec!["the", "dog", "ran"],
        vec!["the", "dog", "ran", "fast"],
        vec!["the", "bird", "flew"],
        vec!["cat", "likes", "fish"],
        vec!["dog", "likes", "bone"],
        vec!["bird", "likes", "seed"],
        vec!["the", "cat", "is", "black"],
        vec!["the", "dog", "is", "brown"],
    ];

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

                ssm.train_full(&training_sentences, 0.0001, &mut model);
                //train(&mut ssm, &tr);
            }
        }
        TrainParam::Embedding => {
            model.train(5, 0.0012, 0.00001);
            model.save_embeddings_txt("./embeddings/seuss.txt").unwrap();
        }
        TrainParam::Both => {
            if args.find_optimal {
                //train_with_optimal(&mut ssm, &tr);
            } else {
                ssm.train_full(&training_sentences, 0.0001, &mut model);
            }
        }
    }
    let out1 = ssm.predict_sequence(&[model.encode_word("the")], 10);
    println!();
    println!("Expected: {:?}", &[4.0, 5.0, 6.0, 7.0]);
    println!("Result: {:?}", out1);
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
