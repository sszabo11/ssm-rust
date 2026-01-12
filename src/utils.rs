use std::{fs, io};

use anyhow::Result;
use serde::{Deserialize, Serialize, de::DeserializeOwned};

pub fn corpus_folder(path: &str) -> String {
    let dir = fs::read_dir(path).unwrap();
    let mut texts = Vec::new();
    for entry in dir {
        let path = entry.unwrap().path();

        if path.is_file() {
            let content = fs::read_to_string(&path).unwrap();
            texts.extend(content.lines().map(|l| l.trim().to_string()));
        }
    }

    texts.join("\n")
}

pub fn corpus_file(path: &str) -> String {
    fs::read_to_string(path).unwrap()
}

pub fn read_csv<R: DeserializeOwned>(path: &str) -> Result<Vec<R>> {
    let mut records = Vec::new();
    let mut rdr = csv::Reader::from_path(path)?;
    for result in rdr.deserialize() {
        let record: R = result?;
        records.push(record);
    }

    Ok(records)
}
