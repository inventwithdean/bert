use std::{env, error::Error, fs, io::Read, process, time::Instant};

use bert::{
    config::BertConfig,
    model::{BertForSequenceClassification, BertForSequenceClassificationConfig},
};
use burn::tensor::{Device, Int, Tensor, TensorData, activation::sigmoid};
use burn_store::{ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};
use flate2::read::GzDecoder;
use serde::Deserialize;
use tokenizers::utils::padding::pad_encodings;
use tokenizers::{Encoding, PaddingParams, Tokenizer};

#[derive(Deserialize)]
struct Entry {
    query: String,
    positive: Vec<String>,
    negative: Vec<String>,
}

impl Entry {
    fn documents(&self) -> Vec<&str> {
        self.positive
            .iter()
            .chain(self.negative.iter())
            .map(String::as_str)
            .collect()
    }
}

struct Stats<T> {
    count: usize,
    min: T,
    p50: T,
    p90: T,
    p99: T,
    max: T,
}

impl<T: Copy + PartialOrd> Stats<T> {
    fn from_values(mut values: Vec<T>) -> Self {
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let at = |p: f64| values[(((values.len() - 1) as f64) * p).round() as usize];
        Self {
            count: values.len(),
            min: values[0],
            p50: at(0.50),
            p90: at(0.90),
            p99: at(0.99),
            max: values[values.len() - 1],
        }
    }
}

#[derive(Default)]
struct Diag {
    padded_tokens: usize,
    real_tokens: usize,
    batches: usize,
}

struct Args {
    data: String,
    device: String,
    weights: String,
    tokenizer: String,
    config: String,
    batch: usize,
    warmup: usize,
    limit: Option<usize>,
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        data: "data/data.jsonl.gz".into(),
        device: "gpu".into(),
        weights: "weights/model.safetensors".into(),
        tokenizer: "weights/tokenizer.json".into(),
        config: "weights/config.json".into(),
        batch: 16,
        warmup: 20,
        limit: None,
    };
    let raw: Vec<String> = env::args().skip(1).collect();
    let mut i = 0;
    while i < raw.len() {
        let value = |i: &mut usize| -> Result<String, String> {
            *i += 1;
            raw.get(*i)
                .cloned()
                .ok_or_else(|| format!("missing value for `{}`", raw[*i - 1]))
        };
        let number = |v: String| v.parse::<usize>().map_err(|e| e.to_string());
        match raw[i].as_str() {
            "--data" => args.data = value(&mut i)?,
            "--device" => args.device = value(&mut i)?,
            "--weights" => args.weights = value(&mut i)?,
            "--tokenizer" => args.tokenizer = value(&mut i)?,
            "--config" => args.config = value(&mut i)?,
            "--batch" => args.batch = number(value(&mut i)?)?,
            "--warmup" => args.warmup = number(value(&mut i)?)?,
            "--limit" => args.limit = Some(number(value(&mut i)?)?),
            "--help" | "-h" => {
                println!("--data --weights --tokenizer --config --batch --warmup --limit");
                process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}")),
        }
        i += 1;
    }
    Ok(args)
}

fn read_entries(path: &str) -> Result<Vec<Entry>, Box<dyn Error + Send + Sync>> {
    let mut decoder = GzDecoder::new(fs::File::open(path)?);
    let mut text = String::new();
    decoder.read_to_string(&mut text)?;
    Ok(text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l))
        .collect::<Result<_, _>>()?)
}

fn rerank(
    model: &BertForSequenceClassification,
    tokenizer: &Tokenizer,
    padding: &PaddingParams,
    device: &Device,
    entry: &Entry,
    batch_size: usize,
    diag: &mut Diag,
) -> Result<Vec<f32>, Box<dyn Error + Send + Sync>> {
    let documents = entry.documents();
    if documents.is_empty() {
        return Ok(Vec::new());
    }

    let pairs: Vec<(&str, &str)> = documents
        .iter()
        .map(|document| (entry.query.as_str(), *document))
        .collect();
    let encodings = tokenizer.encode_batch(pairs, true)?;

    let mut indexed: Vec<(usize, Encoding)> = encodings.into_iter().enumerate().collect();
    indexed.sort_by_key(|(_, encoding)| encoding.len());
    let (order, mut sorted): (Vec<usize>, Vec<Encoding>) = indexed.into_iter().unzip();

    let mut scores = vec![0.0f32; documents.len()];
    for (indices, batch) in order.chunks(batch_size).zip(sorted.chunks_mut(batch_size)) {
        pad_encodings(batch, padding)?;

        let rows = batch.len();
        let seq_len = batch[0].get_ids().len();
        let (mut ids, mut type_ids, mut mask) = (Vec::new(), Vec::new(), Vec::new());
        for encoding in batch.iter() {
            ids.extend(encoding.get_ids().iter().map(|&v| v as i64));
            type_ids.extend(encoding.get_type_ids().iter().map(|&v| v as i64));
            mask.extend(encoding.get_attention_mask().iter().map(|&v| v as i64));
            diag.real_tokens += encoding
                .get_tokens()
                .iter()
                .filter(|token| token.as_str() != "[PAD]")
                .count();
        }
        diag.padded_tokens += rows * seq_len;
        diag.batches += 1;

        let input_ids = Tensor::<2, Int>::from_data(TensorData::new(ids, [rows, seq_len]), device);
        let token_type_ids =
            Tensor::<2, Int>::from_data(TensorData::new(type_ids, [rows, seq_len]), device);
        // flex fills where the bool mask is `true`; tokenizer says 1 = real, 0 = pad.
        let attention_mask =
            Tensor::<2, Int>::from_data(TensorData::new(mask, [rows, seq_len]), device)
                .equal_elem(0);

        let logits = model.forward(input_ids, attention_mask, token_type_ids);
        let batch_scores: Vec<f32> = sigmoid(logits.reshape([rows]))
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        for (&index, score) in indices.iter().zip(batch_scores) {
            scores[index] = score;
        }
    }
    Ok(scores)
}

fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let args = parse_args().map_err(|e| {
        format!(
            "{e}\nusage: bench [--data PATH] [--weights PATH] [--tokenizer PATH] [--config PATH] \
             [--batch N] [--warmup N] [--limit N]"
        )
    })?;

    let device = match args.device.as_str() {
        "gpu" => {
            println!("Using GPU...");
            Device::wgpu(Default::default())
        }
        "cpu" => {
            println!("Using CPU...");
            Device::flex()
        }
        _ => {
            println!("Defaulting to CPU...");
            Device::flex()
        }
    };

    let config: BertConfig = serde_json::from_str(&fs::read_to_string(&args.config)?)?;
    let mut model = BertForSequenceClassificationConfig::new().init(&config, &device);

    let start = Instant::now();
    let mut store = SafetensorsStore::from_file(&args.weights)
        .with_key_remapping(r"\.attention\.self\.", ".attention.self_attention.")
        .with_key_remapping(r"LayerNorm", "layer_norm")
        .with_from_adapter(PyTorchToBurnAdapter);
    let report = model.load_from(&mut store)?;
    // println!("{report}");
    println!(
        "model load: {:.1} ms",
        start.elapsed().as_secs_f64() * 1000.0
    );

    let tokenizer = Tokenizer::from_file(&args.tokenizer)?;
    let padding = PaddingParams::default();

    let mut entries = read_entries(&args.data)?;
    if let Some(limit) = args.limit {
        entries.truncate(limit);
    }
    println!(
        "\ndataset: {} entries, batch_size={} (length-sorted), warmup={}",
        entries.len(),
        args.batch,
        args.warmup
    );

    let mut diag = Diag::default();
    let warm = Instant::now();
    for entry in entries.iter().take(args.warmup) {
        let _ = rerank(
            &model, &tokenizer, &padding, &device, entry, args.batch, &mut diag,
        )?;
    }
    println!(
        "warmup: {} entries in {:.1} ms",
        args.warmup.min(entries.len()),
        warm.elapsed().as_secs_f64() * 1000.0
    );

    diag = Diag::default();
    let mut per_request = Vec::with_capacity(entries.len());
    let mut per_document = Vec::with_capacity(entries.len());
    let mut doc_counts = Vec::with_capacity(entries.len());
    let mut checksum = 0.0f64;

    let total = Instant::now();
    for entry in &entries {
        let count = entry.positive.len() + entry.negative.len();
        let start = Instant::now();
        let scores = rerank(
            &model, &tokenizer, &padding, &device, entry, args.batch, &mut diag,
        )?;
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;

        checksum += scores.iter().map(|&s| s as f64).sum::<f64>();
        per_request.push(elapsed);
        per_document.push(elapsed / count as f64);
        doc_counts.push(count);
    }
    let wall = total.elapsed().as_secs_f64();

    let counts = Stats::from_values(doc_counts.iter().map(|&c| c as f64).collect());
    println!("\nDocuments per entry ({} entries)", counts.count);
    println!(
        "  min: {:.0}  p50: {:.0}  p90: {:.0}  p99: {:.0}  max: {:.0}",
        counts.min, counts.p50, counts.p90, counts.p99, counts.max
    );

    let requests = Stats::from_values(per_request);
    println!("\nPer-request latency ({} entries)", requests.count);
    println!("  min: {:.4}ms", requests.min);
    println!("  p50: {:.4}ms", requests.p50);
    println!("  p90: {:.4}ms", requests.p90);
    println!("  p99: {:.4}ms", requests.p99);
    println!("  max: {:.4}ms", requests.max);

    let documents = Stats::from_values(per_document);
    println!("\nPer-document latency ({} entries)", documents.count);
    println!("  min: {:.4}ms", documents.min);
    println!("  p50: {:.4}ms", documents.p50);
    println!("  p90: {:.4}ms", documents.p90);
    println!("  p99: {:.4}ms", documents.p99);
    println!("  max: {:.4}ms", documents.max);

    let waste = 100.0 * (diag.padded_tokens - diag.real_tokens) as f64 / diag.real_tokens as f64;
    println!("\n--- extra diagnostics ---");
    println!("wall clock:  {wall:.2}s for {} requests", entries.len());
    println!("throughput:  {:.1} requests/s", entries.len() as f64 / wall);
    println!(
        "batches:     {} (mean {:.1} per request)",
        diag.batches,
        diag.batches as f64 / entries.len() as f64
    );
    println!(
        "tokens:      {} real / {} padded -> {waste:.1}% padding waste",
        diag.real_tokens, diag.padded_tokens
    );
    println!("checksum:    {checksum:.4}");
    Ok(())
}
