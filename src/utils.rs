use std::fs;

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
