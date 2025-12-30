use std::collections::HashMap;

use rand::Rng;
use regex::Regex;

use crate::ssm::SSM;

pub fn generate_sequences(
    num_seq_per_step: usize,
    max_step: usize,
    seq_len: usize,
) -> Vec<(Vec<f32>, Vec<f32>)> {
    let mut output: Vec<(Vec<f32>, Vec<f32>)> = Vec::new();

    let mut rng = rand::rng();

    for step in 1..=max_step {
        for i in 0..num_seq_per_step {
            let mut start = rng.random_range(0.0..1000.0);
            let mut seq = Vec::new();
            for _ in 0..seq_len {
                start += step as f32;
                seq.push(start);
            }

            let mut target = seq.clone();
            start += step as f32;
            target.push(start);
            target = target
                .into_iter()
                .rev()
                .take(seq_len)
                .rev()
                .collect::<Vec<f32>>();

            //for _ in 0..seq_len {
            //    target.push(start);
            //}
            output.push((seq, target));
        }
    }

    println!(
        "{:?}",
        output
            .clone()
            .into_iter()
            .take(2)
            .collect::<Vec<(Vec<f32>, Vec<f32>)>>()
    );
    println!("Training len: {}", output.len());
    output
}

pub fn train_with_optimal(ssm: &mut SSM, tr: &Vec<(Vec<f32>, Vec<f32>)>) {
    let mut curr = 0.0000000001;
    let end = 0.0000001;

    let mut result = HashMap::new();
    while curr <= end {
        ssm.reset_state();
        ssm.reset_gradients();
        let epochs = 1000;
        for epoch in 1..=epochs {
            let mut total_loss: f32 = 0.0;
            for (input, target) in tr {
                //let loss = ssm.train(input, target, 0.0001);

                //total_loss += loss;
            }

            if epoch == epochs {
                let avg_loss = total_loss / tr.len() as f32;
                println!("Epoch: {} | Loss: {}", epoch, avg_loss);
                result.insert(total_loss.to_string(), curr);
            }
        }
        curr += 0.0000000001;
    }

    let mut losses = result
        .keys()
        .cloned()
        .map(|k| k.parse::<f32>().unwrap())
        .collect::<Vec<f32>>();
    losses.sort_by(f32::total_cmp);

    println!(
        "Best: {:?} @ learning rate: {}",
        losses[0],
        result.get(&losses[0].to_string()).unwrap()
    );
}
pub fn train(ssm: &mut SSM, tr: &Vec<(Vec<f32>, Vec<f32>)>) {
    ssm.reset_state();
    ssm.reset_gradients();
    let epochs = 1000;
    for epoch in 1..=epochs {
        let mut total_loss: f32 = 0.0;
        for (input, target) in tr {
            //let loss = ssm.train(input, target, 0.0001);

            //total_loss += loss;
        }

        if epoch == epochs {
            let avg_loss = total_loss / tr.len() as f32;
            println!("Epoch: {} | Loss: {}", epoch, avg_loss);
        }
    }
}

pub fn prepare_training_data(corpus: &str) -> Vec<Vec<String>> {
    let mut sentences = Vec::new();

    // Split by sentence-ending punctuation
    let re = Regex::new(r"[.!?]+").unwrap();

    for sentence in re.split(corpus) {
        let sentence = sentence.trim().to_lowercase().clone();

        if sentence.is_empty() {
            continue;
        }

        // Split into words
        let words: Vec<String> = sentence
            .split_whitespace()
            .filter(|w| !w.is_empty())
            .map(|v| v.trim_ascii().to_string())
            .collect();

        // Only keep sentences with 3+ words
        if words.len() >= 3 {
            sentences.push(words);
        }
    }

    sentences
}
