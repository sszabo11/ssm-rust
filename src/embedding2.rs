use anyhow::Result;
use colored::Colorize;
use ndarray::{Array, Array1, Array2, ArrayBase, Axis, Dim, Ix1, Ix2, OwnedRepr, ViewRepr};
use ndarray_npy::{ReadNpyExt, WriteNpyExt};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use rand::{Rng, random_range};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use regex::Regex;
use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::{self, BufRead, BufReader, BufWriter, Read, Write},
    path::Path,
    sync::{Arc, Mutex},
};

pub fn get_chars(corpus: &str, char_len: usize) -> HashMap<String, usize> {
    let mut v: HashMap<String, usize> = HashMap::new();

    println!("len: {}", corpus.len());
    let chars: Vec<char> = corpus.chars().collect();

    //for i in (0..corpus.len()).step_by(char_len) {
    for (i, char) in chars.iter().step_by(char_len).enumerate() {
        if (i + char_len) <= corpus.len() {
            let token: String = (0..char_len).map(|j| chars[i + j]).collect();

            //println!("i: {} | token: {}", i, token);

            if let Some(occurances) = v.get(&token) {
                v.insert(token, *occurances + 1);
            } else {
                v.insert(token, 1);
            }
        }
    }

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

pub struct EmbeddingChar {
    pub output_e: Arc<Mutex<Vector>>,
    pub input_e: Arc<Mutex<Vector>>,
    corpus: String,
    pub vocab: HashMap<String, usize>,  // With occurances
    pub w_to_i: HashMap<String, usize>, // Word to index
    pub i_to_w: HashMap<usize, String>, // Index to word
    pub dim: usize,
    pub vector: Vector,
    sliding_window: usize,
    k: usize,
}

impl EmbeddingChar {
    pub fn new(corpus: String, dim: usize, sliding_window: usize, k: usize) -> Self {
        let vocab = get_chars(&corpus, 1);
        //let vocab = get_sentences(&corpus);
        let filtered_vocab = filter_infrequent_words(&vocab);
        let (w_to_i, i_to_w) = create_vocab_maps(&filtered_vocab);

        let vocab_size = vocab.len();

        println!("Vocab Words: {}", vocab_size);

        Self {
            input_e: Arc::new(Mutex::new(Array::random(
                (vocab_size, dim),
                Uniform::new(-0.1, 0.1).unwrap(),
            ))),
            output_e: Arc::new(Mutex::new(Array::random(
                (vocab_size, dim),
                Uniform::new(-0.1, 0.1).unwrap(),
            ))),
            vector: Array::zeros((vocab_size, dim)),
            corpus,
            dim,
            vocab: filtered_vocab,
            w_to_i,
            k,
            sliding_window,
            i_to_w,
        }
    }
    pub fn predict2(&self, query: &str, len: usize, window: usize, temperature: f32) -> String {
        let input_vector = self.input_e.lock().unwrap();
        let mut prev_tokens = query.to_string();
        let mut response = query.to_string();

        while response.chars().count() <= len {
            //println!("prev: {}", prev_tokens);
            // Compute average embedding of last window chars
            let query_tokens = {
                let mut sum = Array1::<f32>::zeros(self.dim);
                let mut count = 0;
                for c in prev_tokens
                    .chars()
                    .collect::<Vec<char>>()
                    .into_iter()
                    .rev()
                    .take(window)
                    .rev()
                {
                    let c_str = c.to_string();
                    if let Some(idx) = self.w_to_i.get(&c_str) {
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

            // Score all vocab chars
            let mut scores = self
                .vocab
                .keys()
                .map(|char_key| {
                    let idx = self.w_to_i.get(char_key).unwrap();
                    let char_vec = input_vector.row(*idx);
                    let dot: f32 = query_tokens.dot(&char_vec);
                    let mag1 = query_tokens.mapv(|x| x * x).sum().sqrt();
                    let mag2 = char_vec.mapv(|x| x * x).sum().sqrt();
                    let sim = if mag1 == 0.0 || mag2 == 0.0 {
                        0.0
                    } else {
                        dot / (mag1 * mag2)
                    };
                    (sim, char_key.clone())
                })
                .collect::<Vec<(f32, String)>>();

            // Softmax for probabilistic sampling
            let max_score = scores
                .iter()
                .map(|(s, _)| *s)
                .fold(f32::NEG_INFINITY, f32::max);
            let scores_normalized: Vec<f32> = scores
                .iter()
                .map(|(s, _)| ((s - max_score) / temperature).exp())
                .collect();
            let sum_norm: f32 = scores_normalized.iter().sum();
            let probs: Vec<f32> = scores_normalized.iter().map(|s| s / sum_norm).collect();

            // Sample next char
            let mut rng = rand::rng();
            let rand_val: f32 = rng.random();
            let mut cumsum = 0.0;
            let mut new_char = scores[0].1.clone(); // Fallback
            for (i, &prob) in probs.iter().enumerate() {
                cumsum += prob;
                if rand_val < cumsum {
                    new_char = scores[i].1.clone();
                    break;
                }
            }

            scores.sort_by(|a, b| b.0.total_cmp(&a.0).reverse());
            //println!(
            //    "Top: {:?} | Bottom: {:?}",
            //    scores[0],
            //    scores.last().unwrap_or(&(0.0, "".to_string()))
            //);
            //for i in scores.len() - 5..scores.len() {
            //    println!("{}: {}", scores.len() - i, scores[i].1);
            //}

            response += &new_char;
            prev_tokens = response
                .chars()
                .collect::<Vec<char>>()
                .iter()
                .rev()
                .take(window)
                .rev()
                .cloned()
                .collect::<String>()
        }
        response
    }

    pub fn predict(&self, query: &str, len: usize, window: usize, temp: f32) -> String {
        let mut prev_tokens = query.to_lowercase().to_string();
        let mut word = String::from(query);

        //let vec = self.encode_word(&prev_tokens);
        let input_embedding = self.input_e.lock().unwrap();

        while word.len() < len {
            let mut last_chars_vec = Array1::zeros(self.dim);

            for i in 0..window {
                println!("prev: {:?}", prev_tokens);
                let rev = prev_tokens.chars().rev().collect::<Vec<char>>();
                let Some(char) = rev.get(i) else {
                    println!("String too long for window: {}", window);
                    continue;
                };

                let char_str = char.to_string();

                let Some(idx) = self.w_to_i.get(&char_str) else {
                    println!("NO vocab for: '{}' ", char_str);
                    continue;
                };

                let vec = input_embedding.row(*idx);

                //println!("vec: {}", vec);
                last_chars_vec += &vec;
                last_chars_vec /= ((i / window) - 1) as f32;
            }
            let mag = last_chars_vec.mapv(|x| x * x).sum().sqrt();
            if mag > 0.0 {
                last_chars_vec /= mag;
            }

            let scores = self
                .i_to_w
                .iter()
                .map(|(i, w)| {
                    let char_vec = input_embedding.row(*i);
                    let dot: f32 = last_chars_vec.dot(&char_vec);
                    //let mag1 = last_chars_vec.mapv(|x| x * x).sum().sqrt();
                    //let mag2 = char_vec.mapv(|x| x * x).sum().sqrt();

                    //if mag1 == 0.0 || mag2 == 0.0 {
                    //    return (0.0, w.to_string());
                    //}

                    //let res = dot / (mag1 * mag2);
                    //println!("M: {} | {} | {}", res, dot, last_chars_vec);

                    (dot, w.to_string())
                })
                .collect::<Vec<(f32, String)>>();

            //scores.sort_by(|(a, _), (b, _)| b.partial_cmp(a).unwrap());

            let max_score = scores
                .iter()
                .map(|(s, _)| s)
                .fold(f32::NEG_INFINITY, |a, &b| a.max(b));

            let exp_scores: Vec<f32> = scores
                .iter()
                .map(|(s, _)| ((s - max_score) / temp).exp())
                .collect();

            let sum: f32 = exp_scores.iter().sum();
            let probs: Vec<f32> = exp_scores.iter().map(|s| s / sum).collect();

            // Sample from distribution
            let mut rng = rand::rng();
            let rand_val: f32 = rng.random();
            let mut cumsum = 0.0;

            let mut next_char = None;
            for (i, &prob) in probs.iter().enumerate() {
                cumsum += prob;
                if rand_val < cumsum {
                    next_char = Some(scores[i].1.clone());
                    break;
                }
            }

            if let Some(c) = next_char {
                word.push_str(&c);
                prev_tokens.push_str(&c);
            } else {
                break;
            }

            //let w = scores[0].1.clone();

            //prev_tokens.push_str(&w);
            //println!("w {} {}", w, word.len());
            //word.push_str(&w);
        }

        word
    }

    pub fn save_embeddings_npy(&self, path: &str) -> Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        let guard = self.input_e.lock().unwrap();
        guard.write_npy(&mut writer)?;

        Ok(())
    }

    pub fn load_embeddings_npy(&self, path: &str) -> Result<Array2<f32>> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        Ok(Array2::<f32>::read_npy(&mut reader)?)
    }

    pub fn train(&mut self, epochs: usize, init_learning_rate: f32, decay_rate: f32) {
        let max_gradient_norm = 5.0;

        //let mut learning_rate = init_learning_rate;
        let e: Vec<_> = (0..epochs).collect();

        let output_e_clone = Arc::clone(&self.output_e);
        let input_e_clone = Arc::clone(&self.input_e);

        e.iter().for_each(|epoch| {
            //learning_rate = init_learning_rate * (1.0 / (1.0 + decay_rate * *epoch as f32));

            let learning_rate = init_learning_rate;
            println!("Epoch: {}", epoch);

            let output_e = Arc::clone(&output_e_clone);
            let input_e = Arc::clone(&input_e_clone);

            let chars: Vec<String> = self.corpus.chars().map(|c| c.to_string()).collect();

            for (i, target_char) in chars.clone().into_iter().enumerate() {
                if i.is_multiple_of(1000) {
                    println!(
                        "Status from epoch {} at 1000 words. {}/{}",
                        epoch,
                        i,
                        chars.len() - 1
                    )
                }

                let Some(target_idx) = self.w_to_i.get(&target_char) else {
                    continue;
                };

                'window: for w in -(self.sliding_window as isize)..=self.sliding_window as isize {
                    if (i as isize + w) as usize >= chars.len() || w == 0 {
                        continue 'window;
                    };

                    let context_char = chars[(i as isize + w) as usize].clone();
                    let Some(context_idx) = self.w_to_i.get(&context_char) else {
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
                        let pos_error = 1.0 - pos_score;
                        assert!(!pos_error.is_nan());

                        let mut target_update = Array1::<f32>::zeros(input_guard.ncols());

                        // Positive update to target
                        // Learning rate * positive error * output vec of context word
                        target_update +=
                            &(learning_rate * pos_error * &output_guard.row(*context_idx));

                        let norm = target_update.mapv(|v| v * v).sum().sqrt();
                        //println!("norm: {}", norm);
                        if norm > max_gradient_norm {
                            target_update *= max_gradient_norm / norm;
                        }

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
                            let neg_error = 0.0 - neg_score;

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

    pub fn set_words(mut self, corpus: &str) -> Self {
        let words = self.generate_word_embeddings(corpus);

        //let embeddings = words
        //    .iter()
        //    .map(|w| w.1.clone())
        //    .collect::<Vec<Array1<f32>>>();

        //let e = Array2::from_shape_vec((words.len(), dim), embedding).unwrap();
        let width = words.len();

        let mut vocab = HashMap::new();
        let mut arr = Array2::zeros((width, self.dim));
        for (i, (word, vec)) in words.iter().enumerate() {
            arr.push(Axis(0), vec.view()).unwrap();

            vocab.insert(word.clone(), i);
        }

        println!("arr d: {:?}", arr.dim());
        let mut input = self.input_e.lock().unwrap();
        *input = arr;

        self.vocab = vocab;
        drop(input);

        //let filtered_vocab = filter_infrequent_words(&vocab);
        //let (w_to_i, i_to_w) = create_vocab_maps(&filtered_vocab);
        //self.w_to_i = w_to_i;
        //self.i_to_w = i_to_w;

        //let vocab_size = vocab.len();

        //println!("Vocab Words: {}", vocab_size);
        self
    }

    pub fn generate_word_embeddings(&self, corpus: &str) -> HashMap<String, Array1<f32>> {
        let words: Vec<String> = corpus
            .split_whitespace()
            .map(|w| w.replace("\"", ""))
            .collect();

        let idxs: Vec<Vec<(usize, char)>> = words
            .into_iter()
            .map(|word| {
                word.chars()
                    .map(|c| (*self.w_to_i.get(&c.to_string()).unwrap(), c))
                    .collect::<Vec<(usize, char)>>()
            })
            .collect();

        let input_embedding = self.input_e.lock().unwrap();

        let mut map = HashMap::new();
        for char_idxs in idxs.iter() {
            let mut sum = Array1::<f32>::zeros(self.dim);
            let mut word = String::new();

            for (idx, char) in char_idxs.iter() {
                let v = input_embedding.row(*idx);
                word.push(*char);
                sum += &v;
            }

            map.insert(word, sum);
        }

        map
    }

    pub fn decode_word(&self, vec: Array2<f32>, temp: f32) -> String {
        let input_embedding = self.input_e.lock().unwrap();

        let mut word = String::new();

        println!("v d {:?} {}", vec.dim(), vec.nrows());
        for char in vec.rows() {
            let mut scores = self
                .i_to_w
                .iter()
                .map(|(i, w)| {
                    let word_vec = input_embedding.row(*i);
                    let dot: f32 = char.dot(&word_vec);
                    let mag1 = char.mapv(|x| x * x).sum().sqrt();
                    let mag2 = word_vec.mapv(|x| x * x).sum().sqrt();

                    if mag1 == 0.0 || mag2 == 0.0 {
                        return (0.0, w.to_string());
                    }

                    (dot / (mag1 * mag2), w.to_string())
                })
                .collect::<Vec<(f32, String)>>();

            //let mut scores_temp: Vec<f32> = scores.iter().map(|(s, _)| *s).collect();
            //scores_temp.sort_by(|a, b| b.total_cmp(a));

            //let scores_temp: Vec<f32> = scores.iter().map(|(s, _)| (s / temp).exp()).collect();

            //let sum: f32 = scores_temp.iter().sum();
            //let probs: Vec<f32> = scores_temp.iter().map(|s| s / sum).collect();
            ////println!("sc: {:?} {}", scores_temp, sum);

            //let mut rng = rand::rng();
            //let rand_val: f32 = rng.random();
            //let mut cumsum = 0.0;

            //println!("PR: {:?}", probs);

            //for (i, &prob) in probs.iter().enumerate() {
            //    cumsum += prob;
            //    if rand_val < cumsum {
            //        let w = scores[i].1.clone();
            //        //println!("w2 {}", w);
            //        word.push_str(&w);
            //    }
            //}

            // Fallback to highest similarity
            scores.sort_by(|(a, _), (b, _)| b.partial_cmp(a).unwrap());

            let w = scores[0].1.clone();

            //println!("w {} {}", w, word.len());
            word.push_str(&w);
        }

        word

        //scores.sort_by(|(v1, s1), (v2, s2)| v1.total_cmp(v2));

        //println!("Sco: {:?} {:?}", scores[0], scores.last().unwrap());
        //for i in scores.len() - 5..scores.len() {
        //    println!("{}: {}", scores.len() - i, scores[i].1);
        //}

        //scores.last().unwrap().1.clone()
    }
    pub fn decode_word2(&self, vec: Array2<f32>) -> String {
        let input_embedding = self.input_e.lock().unwrap();
        let mut word = String::new();

        for char_embedding in vec.rows() {
            let (best_char, _best_score) = self
                .i_to_w
                .iter()
                .map(|(i, w)| {
                    let word_vec = input_embedding.row(*i);
                    let similarity = char_embedding.dot(&word_vec);
                    (w.as_str(), similarity)
                })
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap();

            word.push_str(best_char);
        }

        word
    }
    pub fn encode_word(&self, word: &str) -> Array2<f32> {
        let input_embedding = self.input_e.lock().unwrap();

        //let mut sum = Array1::<f32>::zeros(self.dim);

        let mut encoding = Array2::<f32>::zeros((word.len(), self.dim));
        println!("ed {:?}", encoding.dim());

        for (char_i, char) in word.chars().enumerate() {
            let idx = self.w_to_i[&char.to_string()];
            let vec = input_embedding.row(idx);

            let mut c = encoding.row_mut(char_i);
            c += &vec;

            //sum += &vec;
        }

        //let encoded = sum / word.len() as f32;

        // NORMALIZE
        //let magnitude = encoded.mapv(|x| x * x).sum().sqrt();
        //if magnitude > 0.0 {
        //    encoded /= magnitude;
        //}

        //Array1::from_vec(encoded.to_vec())

        encoding
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
