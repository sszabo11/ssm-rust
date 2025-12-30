use std::{
    collections::HashMap,
    fs::File,
    io::{self, BufRead, BufReader, BufWriter, Read, Write},
    path::Path,
    sync::{Arc, Mutex, MutexGuard},
};

use colored::Colorize;
use ndarray::{Array, Array1, Array2, ArrayBase, Axis, Dim, Ix1, Ix2, OwnedRepr, ViewRepr};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use rand::{Rng, random_range};
use rayon::{
    iter::{
        IntoParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelBridge,
        ParallelIterator,
    },
    str::ParallelString,
};
use regex::Regex;

pub fn get_vocab_n(corpus: &str, word_len: usize) -> HashMap<String, usize> {
    let mut v: HashMap<String, usize> = HashMap::new();

    //for (i, line) in corpus.iter().enumerate() {
    let words: Vec<&str> = corpus.split_whitespace().collect();

    for i in (0..words.len()).step_by(word_len) {
        let words: Vec<String> = (0..word_len)
            .filter_map(|j| {
                if (i + j) < words.len() {
                    Some(tokenize_word(words[i + j]))
                } else {
                    None
                }
            })
            .collect();

        let word = words.join(" ");
        //let word: String = word
        //    .chars()
        //    .filter(|c| !c.is_ascii_punctuation())
        //    .map(|c| c.to_ascii_lowercase())
        //    .collect();

        if let Some(occurances) = v.get(&word) {
            v.insert(word, *occurances + 1);
        } else {
            v.insert(word, 1);
        }
    }
    //}

    v
}
pub fn get_vocab(corpus: &str) -> HashMap<String, usize> {
    let mut v: HashMap<String, usize> = HashMap::new();

    //for (i, line) in corpus.iter().enumerate() {
    let words: Vec<&str> = corpus.split_whitespace().collect();

    for (j, word) in words.iter().enumerate() {
        let word = tokenize_word(word);
        //let word: String = word
        //    .chars()
        //    .filter(|c| !c.is_ascii_punctuation())
        //    .map(|c| c.to_ascii_lowercase())
        //    .collect();

        if let Some(occurances) = v.get(&word) {
            v.insert(word, *occurances + 1);
        } else {
            v.insert(word, 1);
        }
    }
    //}

    v
}

fn filter_infrequent_words(vocab: &HashMap<String, usize>) -> HashMap<String, usize> {
    let mut new_v = HashMap::new();
    for word in vocab.iter() {
        if *word.1 > 0 {
            new_v.insert(word.0.to_string(), *word.1);
        }
    }

    new_v
}

fn create_vocab_maps(
    v: &HashMap<String, usize>,
) -> (HashMap<String, usize>, HashMap<usize, String>) {
    let mut w_to_i: HashMap<String, usize> = HashMap::new();
    let mut i_to_w: HashMap<usize, String> = HashMap::new();

    for (i, word) in v.keys().enumerate() {
        w_to_i.insert(word.to_string(), i);
        i_to_w.insert(i, word.to_string());
    }

    (w_to_i, i_to_w)
}

type Vector = ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>, f32>;

pub struct Embedding {
    pub output_e: Arc<Mutex<Vector>>,
    pub input_e: Arc<Mutex<Vector>>,
    corpus: String,
    pub vocab: HashMap<String, usize>,  // With occurances
    pub w_to_i: HashMap<String, usize>, // Word to index
    i_to_w: HashMap<usize, String>,     // Index to word
    pub dim: usize,
    sliding_window: usize,
    k: usize,
}

fn get_sentences(corpus: &str) -> HashMap<String, usize> {
    let mut map: HashMap<String, usize> = HashMap::new();

    corpus.lines().for_each(|s| {
        println!("s {:?}", s);
        let (q, a) = s.split_once('\t').unwrap();
        map.insert(tokenize_word(q), 1);
        map.insert(tokenize_word(a), 1);
        //(tokenize_word(q).trim_ascii_start().to_string(), 1)
    });

    println!("{}", map.len());
    map
}
impl Embedding {
    pub fn new(corpus: String, dim: usize, sliding_window: usize, k: usize) -> Self {
        let vocab = get_vocab_n(&corpus, 1);
        //let vocab = get_sentences(&corpus);
        let filtered_vocab = filter_infrequent_words(&vocab);
        let (w_to_i, i_to_w) = create_vocab_maps(&filtered_vocab);

        let vocab_size = vocab.len();

        println!("Vocab Words: {}", vocab_size);

        //println!("vocab: {:?}", vocab);
        Self {
            input_e: Arc::new(Mutex::new(Array::random(
                (vocab_size, dim),
                Uniform::new(-0.5, 0.5).unwrap(),
            ))),
            output_e: Arc::new(Mutex::new(Array::random(
                (vocab_size, dim),
                Uniform::new(-0.5, 0.5).unwrap(),
            ))),
            corpus,
            dim,
            vocab: filtered_vocab,
            w_to_i,
            k,
            sliding_window,
            i_to_w,
        }
    }

    pub fn sentences2(&self) -> HashMap<String, Array<f32, Ix1>> {
        let words: Vec<String> = self.vocab.keys().cloned().collect();

        let mut map = HashMap::new();
        println!("words: {:?}", words.len());

        let input_vector = self.input_e.lock().unwrap();

        let re = Regex::new(r"[.?!]\s+|$").unwrap();

        for line in self.corpus.lines() {
            println!("line: {}", line);
            for s in re.split(line) {
                println!("{:?}", s);
                //sentences_list.push(s.to_string());

                let words: Vec<&str> = s.split_whitespace().collect();

                let mut sum = Array1::<f32>::zeros(self.dim);
                let mut count = 0;
                for word in words.iter() {
                    if let Some(idx) = self.w_to_i.get(*word) {
                        sum += &input_vector.row(*idx);
                        count += 1;
                    };
                }

                let result = if count > 0 {
                    sum / count as f32
                } else {
                    Array1::<f32>::zeros(self.dim)
                };

                map.insert(s.to_string(), result);

                //sentence_e.push(Axis(0), result.view()).unwrap();
            }

            //let q_words: Vec<&str> = q.split_whitespace().collect();
            //let a_words: Vec<&str> = a.split_whitespace().collect();

            //let mut sum = Array1::<f32>::zeros(self.dim);
            //let mut count = 0;
            //for word in q_words.iter() {
            //    if let Some(idx) = self.w_to_i.get(*word) {
            //        sum += &input_vector.row(*idx);
            //        count += 1;
            //    };
            //}
        }
        map
    }

    pub fn predict(&self, query: &str, len: usize, window: usize) -> String {
        println!("Predicting...");

        let input_vector = self.input_e.lock().unwrap();
        let mut prev_tokens = query.to_string();
        let mut response = query.to_lowercase();

        println!("v le {}", self.vocab.len());

        while response.len() <= len {
            println!("prev: {}", prev_tokens);
            let mut scores = self
                .vocab
                .keys()
                .map(|word| {
                    //let query_tokens = self.encode_question(&prev_word);
                    let query_tokens = {
                        let mut sum = Array1::<f32>::zeros(self.dim);
                        let mut count = 0;

                        for word in prev_tokens.split_whitespace() {
                            //println!("W: {}", word);
                            if let Some(idx) = self.w_to_i.get(&word.to_lowercase()) {
                                sum += &input_vector.row(*idx);
                                count += 1;
                            }
                        }
                        if count > 0 {
                            sum / count as f32
                        } else {
                            Array1::<f32>::zeros(self.dim)
                        }
                    };

                    let word_idx = self.w_to_i.get(word).unwrap();
                    let word_vec = input_vector.row(*word_idx);

                    let dot: f32 = query_tokens.dot(&word_vec);
                    let mag1 = query_tokens.mapv(|x| x * x).sum().sqrt();
                    let mag2 = word_vec.mapv(|x| x * x).sum().sqrt();

                    //println!("{} {}", mag1, mag2);
                    if mag1 == 0.0 || mag2 == 0.0 {
                        return (0.0, word.to_string());
                    }

                    (dot / (mag1 * mag2), word.to_string())
                })
                .collect::<Vec<(f32, String)>>();

            scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            //scores.sort_by(|(v1, s1), (v2, s2)| v1.total_cmp(v2));
            let new_word = scores[0].1.clone();

            //println!("Sco: {:?} {:?}", scores[0], scores.last().unwrap());
            for i in scores.len() - 5..scores.len() {
                println!("{}: {}", scores.len() - i, scores[i].1);
            }

            response += " ";
            response += &new_word;

            prev_tokens = response
                .split_whitespace()
                .collect::<Vec<&str>>()
                .iter()
                .rev()
                .take(window)
                .rev()
                .cloned()
                .collect::<Vec<&str>>()
                .join(" ");
        }

        response
    }
    pub fn sentences(&self) -> HashMap<String, Array<f32, Ix1>> {
        let words: Vec<String> = self.vocab.keys().cloned().collect();

        //let mut sentence_e = Array2::<f32>::zeros((2000, self.dim));
        //let mut sentences_list = Vec::new();
        let mut map = HashMap::new();
        println!("words: {:?}", words.len());

        let qa: Vec<&str> = self.corpus.lines().collect();

        let input_vector = self.input_e.lock().unwrap();

        for line in qa.iter() {
            println!("line: {}", line);
            for s in line.split('\t') {
                println!("{:?}", s);
                //sentences_list.push(s.to_string());

                let words: Vec<&str> = s.split_whitespace().collect();

                let mut sum = Array1::<f32>::zeros(self.dim);
                let mut count = 0;
                for word in words.iter() {
                    if let Some(idx) = self.w_to_i.get(*word) {
                        sum += &input_vector.row(*idx);
                        count += 1;
                    };
                }

                let result = if count > 0 {
                    sum / count as f32
                } else {
                    Array1::<f32>::zeros(self.dim)
                };

                map.insert(s.to_string(), result);

                //sentence_e.push(Axis(0), result.view()).unwrap();
            }

            //let q_words: Vec<&str> = q.split_whitespace().collect();
            //let a_words: Vec<&str> = a.split_whitespace().collect();

            //let mut sum = Array1::<f32>::zeros(self.dim);
            //let mut count = 0;
            //for word in q_words.iter() {
            //    if let Some(idx) = self.w_to_i.get(*word) {
            //        sum += &input_vector.row(*idx);
            //        count += 1;
            //    };
            //}
        }
        map
    }

    pub fn embed(&self) {
        let vocab_size = self.vocab.len();

        //let mut a = Array::random((vocab_size, self.dim), Uniform::new(0.0, 1.0).unwrap());

        println!("Vocab: {:?}", self.w_to_i);

        println!("Vocab size: {}", vocab_size);

        let sliding_window = 1;

        //for line in self.corpus.iter() {
        let words: Vec<&str> = self.corpus.split_whitespace().collect();

        for i in 0..words.len() {
            if i > 1 {
                let word = tokenize_word(words[i]);

                for w in 1..=sliding_window {
                    let word1 = tokenize_word(words[i - w]);
                    let word2 = tokenize_word(words[i + w]);
                    // self.compute_dot_product(&word, &word1);
                    // self.compute_dot_product(&word, &word2);
                }
            }
        }
        //}
    }

    pub fn sentence(&mut self, epochs: usize, learning_rate: f32) {
        let max_gradient_norm = 5.0;
        let e: Vec<_> = (0..epochs).collect();

        let output_e_clone = Arc::clone(&self.output_e);
        let input_e_clone = Arc::clone(&self.input_e);
        println!("c: {} {:?}", self.vocab.len(), self.vocab.keys());
        let sentences: Vec<String> = self.vocab.keys().cloned().collect();

        println!("sentec: {:?}", sentences.len());
        e.iter().for_each(|epoch| {
            println!("Epoch: {}", epoch);

            let output_e = Arc::clone(&output_e_clone);
            let input_e = Arc::clone(&input_e_clone);

            let qa: Vec<&str> = self.corpus.lines().collect();

            for i in 0..qa.len() {
                //println!("{:?}", qa);
                let (q, a) = qa[i].split_once('\t').unwrap();
                if i.is_multiple_of(100) {
                    println!(
                        "Status from epoch {} at 100 sentences. {}/{}",
                        epoch,
                        i,
                        sentences.len() - 1
                    )
                }

                let target_sentence = tokenize_word(q);
                let Some(target_idx) = self.w_to_i.get(&target_sentence) else {
                    continue;
                };

                let context_word = tokenize_word(a);
                let Some(context_idx) = self.w_to_i.get(&context_word) else {
                    continue;
                };

                {
                    let mut input_guard = input_e.lock().unwrap();
                    let mut output_guard = output_e.lock().unwrap();

                    // Similarity
                    let pos_dot_score = input_guard
                        .row(*target_idx)
                        .dot(&output_guard.row(*context_idx));

                    assert!(
                        !pos_dot_score.is_nan(),
                        "{} \n{}",
                        input_guard.row(*target_idx),
                        &output_guard.row(*context_idx)
                    );
                    // Clamps 0-1
                    let pos_score = sigmoid(pos_dot_score);
                    assert!(!pos_score.is_nan());

                    // Error (want to be lower)
                    let pos_error = pos_score - 1.0;
                    assert!(!pos_error.is_nan());

                    let mut target_update = Array1::<f32>::zeros(input_guard.ncols());

                    let norm = target_update.mapv(|v| v * v).sum().sqrt();
                    if norm > max_gradient_norm {
                        target_update *= max_gradient_norm / norm;
                    }
                    // Positive update to target
                    // Learning rate * positive error * output vec of context word
                    target_update += &(learning_rate * pos_error * &output_guard.row(*context_idx));

                    // We do k negative samples at random
                    for _ in 0..self.k {
                        let neg_context_idx = random_range(0..self.i_to_w.len());
                        assert!(neg_context_idx < self.i_to_w.len());

                        let neg_dot_score = input_guard
                            .row(*target_idx)
                            .dot(&output_guard.row(neg_context_idx));
                        assert!(
                            !neg_dot_score.is_nan(),
                            "{} \n{}",
                            input_guard.row(*target_idx),
                            &output_guard.row(neg_context_idx)
                        );

                        let neg_score = sigmoid(neg_dot_score);
                        assert!(!neg_score.is_nan());
                        let neg_error = neg_score;

                        // How much to nudge
                        let update_factor = learning_rate * neg_error;

                        assert!(!update_factor.is_nan());

                        target_update += &(update_factor * &output_guard.row(neg_context_idx));

                        let mut o_n = output_guard.row_mut(neg_context_idx);

                        // Update negative output vector (push away)
                        o_n += &(update_factor * &input_guard.row(*target_idx));
                    }

                    // Apply all accumalted updates to target input vector
                    let mut mut_target_input = input_guard.row_mut(*target_idx);
                    mut_target_input += &target_update;

                    // Updates positive output vector (pull towards target)
                    let pos_update_factor = learning_rate * pos_error;

                    let mut mut_context_output = output_guard.row_mut(*context_idx);
                    mut_context_output += &(pos_update_factor * &input_guard.row(*target_idx));
                }
                //}
            }
            //}
        });
    }

    pub fn train(&mut self, epochs: usize, init_learning_rate: f32, decay_rate: f32) {
        let max_gradient_norm = 5.0;

        let mut learning_rate = init_learning_rate;
        let e: Vec<_> = (0..epochs).collect();

        let output_e_clone = Arc::clone(&self.output_e);
        let input_e_clone = Arc::clone(&self.input_e);
        let words: Vec<String> = self.vocab.keys().cloned().collect();

        e.iter().for_each(|epoch| {
            learning_rate = init_learning_rate * (1.0 / (1.0 + decay_rate * *epoch as f32));
            //if epoch.is_multiple_of(5) && *epoch != 0
            //    println!("Drawing");
            //    let data = self.input_e.lock().unwrap();
            //    draw2(&words, data.clone(), 1, 10).unwrap();
            //    drop(data);
            //}

            println!("Epoch: {}", epoch);

            let output_e = Arc::clone(&output_e_clone);
            let input_e = Arc::clone(&input_e_clone);

            let words: Vec<&str> = self.corpus.split_whitespace().collect();

            for i in 0..words.len() {
                if i.is_multiple_of(1000) {
                    println!(
                        "Status from epoch {} at 1000 words. {}/{}",
                        epoch,
                        i,
                        words.len() - 1
                    )
                }

                let target_word = tokenize_word(words[i]);
                let Some(target_idx) = self.w_to_i.get(&target_word) else {
                    continue;
                };

                'window: for w in -(self.sliding_window as isize)..=self.sliding_window as isize {
                    if (i as isize + w) as usize >= words.len() || w == 0 {
                        continue 'window;
                    };

                    let context_word = tokenize_word(words[(i as isize + w) as usize]);
                    let Some(context_idx) = self.w_to_i.get(&context_word) else {
                        continue;
                    };

                    {
                        let mut input_guard = input_e.lock().unwrap();
                        let mut output_guard = output_e.lock().unwrap();

                        // Similarity
                        let pos_dot_score = input_guard
                            .row(*target_idx)
                            .dot(&output_guard.row(*context_idx));

                        assert!(
                            !pos_dot_score.is_nan(),
                            "{} \n{}",
                            input_guard.row(*target_idx),
                            &output_guard.row(*context_idx)
                        );
                        // Clamps 0-1
                        let pos_score = sigmoid(pos_dot_score);
                        assert!(!pos_score.is_nan());

                        // Error (want to be lower)
                        let pos_error = pos_score - 1.0;
                        assert!(!pos_error.is_nan());

                        let mut target_update = Array1::<f32>::zeros(input_guard.ncols());

                        let norm = target_update.mapv(|v| v * v).sum().sqrt();
                        if norm > max_gradient_norm {
                            target_update *= max_gradient_norm / norm;
                        }
                        // Positive update to target
                        // Learning rate * positive error * output vec of context word
                        target_update +=
                            &(learning_rate * pos_error * &output_guard.row(*context_idx));

                        // We do k negative samples at random
                        for _ in 0..self.k {
                            let neg_context_idx = random_range(0..self.i_to_w.len());
                            assert!(neg_context_idx < self.i_to_w.len());

                            let neg_dot_score = input_guard
                                .row(*target_idx)
                                .dot(&output_guard.row(neg_context_idx));
                            assert!(
                                !neg_dot_score.is_nan(),
                                "{} \n{}",
                                input_guard.row(*target_idx),
                                &output_guard.row(neg_context_idx)
                            );

                            let neg_score = sigmoid(neg_dot_score);
                            assert!(!neg_score.is_nan());
                            let neg_error = neg_score;

                            // How much to nudge
                            let update_factor = learning_rate * neg_error;

                            assert!(!update_factor.is_nan());

                            target_update += &(update_factor * &output_guard.row(neg_context_idx));

                            let mut o_n = output_guard.row_mut(neg_context_idx);

                            // Update negative output vector (push away)
                            o_n += &(update_factor * &input_guard.row(*target_idx));
                        }

                        // Apply all accumalted updates to target input vector
                        let mut mut_target_input = input_guard.row_mut(*target_idx);
                        mut_target_input += &target_update;

                        // Updates positive output vector (pull towards target)
                        let pos_update_factor = learning_rate * pos_error;

                        let mut mut_context_output = output_guard.row_mut(*context_idx);
                        mut_context_output += &(pos_update_factor * &input_guard.row(*target_idx));
                    }
                }
                //}
            }
            //}
        });
    }

    pub fn decode_word(&self, vec: Array1<f32>, temp: f32) -> String {
        let input_embedding = self.input_e.lock().unwrap();

        let mut scores = self
            .vocab
            .iter()
            .map(|(w, i)| {
                let word_vec = input_embedding.row(*i);
                let dot: f32 = vec.dot(&word_vec);
                let mag1 = vec.mapv(|x| x * x).sum().sqrt();
                let mag2 = word_vec.mapv(|x| x * x).sum().sqrt();

                if mag1 == 0.0 || mag2 == 0.0 {
                    return (0.0, w.to_string());
                }

                (dot / (mag1 * mag2), w.to_string())
            })
            .collect::<Vec<(f32, String)>>();

        let scores_temp: Vec<f32> = scores.iter().map(|(s, _)| (s / temp).exp()).collect();

        let sum: f32 = scores_temp.iter().sum();
        let probs: Vec<f32> = scores_temp.iter().map(|s| s / sum).collect();
        println!("sc: {:?} {}", scores_temp, sum);

        let mut rng = rand::rng();
        let rand_val: f32 = rng.random();
        let mut cumsum = 0.0;

        println!("PR: {:?}", probs);

        for (i, &prob) in probs.iter().enumerate() {
            cumsum += prob;
            if rand_val < cumsum {
                return scores[i].1.clone();
            }
        }

        // Fallback to highest similarity
        scores.sort_by(|(a, _), (b, _)| b.partial_cmp(a).unwrap());
        scores[0].1.clone()

        //scores.sort_by(|(v1, s1), (v2, s2)| v1.total_cmp(v2));

        //println!("Sco: {:?} {:?}", scores[0], scores.last().unwrap());
        //for i in scores.len() - 5..scores.len() {
        //    println!("{}: {}", scores.len() - i, scores[i].1);
        //}

        //scores.last().unwrap().1.clone()
    }
    pub fn encode_word(&self, word: &str) -> Array1<f32> {
        let input_embedding = self.input_e.lock().unwrap();

        let word: String = word.chars().filter(|c| !c.is_ascii_punctuation()).collect();

        let i_idx = self
            .w_to_i
            .get(&word)
            .ok_or(format!("No word found for: '{}'", word))
            .unwrap_or(&0);

        let mut encoded_input = input_embedding.row(*i_idx).to_owned();

        // NORMALIZE
        let magnitude = encoded_input.mapv(|x| x * x).sum().sqrt();
        if magnitude > 0.0 {
            encoded_input /= magnitude;
        }

        Array1::from_vec(encoded_input.to_vec())
    }
    pub fn encode_data(&self, data: Vec<Vec<&'static str>>) -> Vec<Vec<f32>> {
        let input_embedding = self.input_e.lock().unwrap();

        let mut output = Vec::new();
        for sentence in data.into_iter() {
            for word in sentence.iter() {
                let i_idx = self
                    .w_to_i
                    .get(*word)
                    .ok_or(format!("No word found for: {}", word))
                    .unwrap();
                let encoded_input = &input_embedding.row(*i_idx);

                //let t_idx = self.w_to_i.get(&target).expect("No word found");
                //let encoded_target = &input_embedding.row(*t_idx);

                output.push(encoded_input.to_vec());
            }
        }

        output
    }
    pub fn encode_question(&self, question: &str) -> Array<f32, Ix1> {
        let input_embedding = self.input_e.lock().unwrap();
        let mut sum = Array1::<f32>::zeros(self.dim);
        let mut count = 0;

        for word in question.split_whitespace() {
            if let Some(idx) = self.w_to_i.get(word) {
                sum += &input_embedding.row(*idx);
                count += 1;
            }
        }
        if count > 0 {
            sum / count as f32
        } else {
            Array1::<f32>::zeros(self.dim)
        }
    }

    pub fn find_answer(
        &self,
        sentences: &HashMap<String, Array<f32, Ix1>>,
        q_v: Array<f32, Ix1>,
    ) -> String {
        let mut scores = sentences
            .iter()
            .map(|(s, v)| {
                let dot: f32 = q_v.dot(v);
                let mag1 = q_v.mapv(|x| x * x).sum().sqrt();
                let mag2 = v.mapv(|x| x * x).sum().sqrt();

                //println!("{} {}", mag1, mag2);
                if mag1 == 0.0 || mag2 == 0.0 {
                    return (0.0, s.to_string());
                }

                (dot / (mag1 * mag2), s.to_string())
            })
            .collect::<Vec<(f32, String)>>();

        scores.sort_by(|(v1, s1), (v2, s2)| v1.total_cmp(v2));

        println!("Sco: {:?} {:?}", scores[0], scores.last().unwrap());
        for i in scores.len() - 5..scores.len() {
            println!("{}: {}", scores.len() - i, scores[i].1);
        }

        scores.last().unwrap().1.clone()
    }

    pub fn save_embedding(&self) {
        //let mut file = File::create("./graph/embedding.txt").unwrap();
    }

    //pub fn load_embeddings_txt2(&self, path: &str) -> std::io::Result<()> {
    //    let buf = String::new();
    //    let file = File::open(path)?.read_to_string(buf);
    //    let mut writer = BufReader::new(file);

    //    let mut words: Vec<_> = self.w_to_i.iter().collect();
    //    words.sort_by(|a, b| a.0.cmp(b.0));

    //    let arr = Array2::<f32>::zeros((2000, self.dim));

    //    for (word, &idx) in words {
    //        arr.push(Axis(0), vec)
    //        write!(&mut writer, "{}", word)?;
    //        for val in vec.iter() {
    //            write!(&mut writer, " {}", val)?;
    //        }
    //        writeln!(&mut writer)?;
    //    }

    //    Ok(())
    //}
    pub fn load_embeddings_txt(
        &self,
        path: &str,
    ) -> Result<
        (HashMap<String, usize>, HashMap<usize, String>, Array2<f32>),
        Box<dyn std::error::Error>,
    > {
        let path = Path::new(path);
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        let mut line = String::new();
        reader.read_line(&mut line)?;
        let header: Vec<&str> = line.split_whitespace().collect();
        let vocab_size: usize = header[0].parse()?;
        let dim: usize = header[1].parse()?;

        let mut w_to_i: HashMap<String, usize> = HashMap::with_capacity(vocab_size);
        let mut i_to_w: HashMap<usize, String> = HashMap::with_capacity(vocab_size);
        let mut matrix = Array2::<f32>::zeros((vocab_size, dim));

        for (idx, line_result) in reader.lines().enumerate() {
            let line = line_result?;
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.is_empty() {
                continue;
            }

            let word = parts[0].to_string();
            println!("p:{:?}", parts);
            let vec: Vec<f32> = parts[1..]
                .iter()
                .map(|s| s.parse::<f32>().unwrap())
                .collect();

            assert_eq!(vec.len(), dim, "Dimension mismatch at word: {}", word);

            w_to_i.insert(word.clone(), idx);
            i_to_w.insert(idx, word);
            matrix.row_mut(idx).assign(&ndarray::Array1::from(vec));
        }

        Ok((w_to_i, i_to_w, matrix))
    }

    pub fn set_input(&mut self, input: Array<f32, Ix2>) {
        let mut i = self.input_e.lock().unwrap();

        *i = input
    }

    pub fn question_answer_loop(&self, sentences: &HashMap<String, Array<f32, Ix1>>) {
        loop {
            let mut input = String::new();
            io::stdout().flush().expect("Failed to flush stdout");

            io::stdin()
                .read_line(&mut input)
                .expect("Failed to read line");

            let input = input.trim();
            println!("\n{}", input.yellow());

            if input == "quit" || input == "exit" {
                println!("Exiting program...");
                break;
            }

            if input.is_empty() {
                println!("You didn't type anything. Try again.");
            } else {
                let q_v = self.encode_question(input);
                let answer = self.find_answer(sentences, q_v);
                println!("{}", answer.green());
            }
        }
    }
    pub fn save_vocab(&self, path: &str) -> std::io::Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        for (word, i) in self.vocab.iter() {
            write!(&mut writer, "{}", word)?;
            writeln!(&mut writer)?;
        }

        Ok(())
    }
    pub fn save_embeddings_txt(&self, path: &str) -> std::io::Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        writeln!(&mut writer, "{} {}", self.w_to_i.len(), self.dim)?;

        let guard = self.input_e.lock().unwrap();

        let mut words: Vec<_> = self.w_to_i.iter().collect();
        words.sort_by(|a, b| a.0.cmp(b.0));

        for (word, &idx) in words {
            let vec = guard.row(idx);
            write!(&mut writer, "{}", word)?;
            for val in vec.iter() {
                write!(&mut writer, " {}", val)?;
            }
            writeln!(&mut writer)?;
        }

        Ok(())
    }

    pub fn cosine_similarity(&self, word1: &str, word2: &str) -> Option<f32> {
        let input_e = self.input_e.lock().unwrap();

        let word1_idx = self.w_to_i.get(word1).unwrap();
        let word2_idx = self.w_to_i.get(word2).unwrap();

        let word1_vec = input_e.row(*word1_idx);
        let word2_vec = input_e.row(*word2_idx);

        let dot: f32 = word1_vec.dot(&word2_vec);
        let mag1 = word1_vec.mapv(|x| x * x).sum().sqrt();
        let mag2 = word2_vec.mapv(|x| x * x).sum().sqrt();

        if mag1 == 0.0 || mag2 == 0.0 {
            return Some(0.0); // or None if you prefer
        }

        let res = dot / (mag1 * mag2);

        //println!("dot: {} | mag1: {} | mag2: {}", dot, mag1, mag2);
        println!("{} & {} = {}", word1, word2, res);
        Some(res)
    }
}
fn magnitude(input: &[f32]) -> f32 {
    let s: f32 = input.iter().map(|v| v.powf(2.0)).sum();

    s.sqrt()
}

fn tokenize_word(word: &str) -> String {
    let word: String = word
        .chars()
        .filter(|c| !c.is_ascii_punctuation())
        .map(|c| c.to_ascii_lowercase())
        .collect();
    word
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
