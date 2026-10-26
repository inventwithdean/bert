use std::{error::Error, fs, time::Instant};

use bert::{
    config::BertConfig,
    model::{BertForSequenceClassification, BertForSequenceClassificationConfig},
};
use burn::tensor::{Device, Int, Tensor, TensorData, activation::sigmoid};
use burn_store::{ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};
use tokenizers::{PaddingParams, Tokenizer};

const WEIGHTS: &str = "weights/model.safetensors";
const TOKENIZER: &str = "weights/tokenizer.json";
const CONFIG: &str = "weights/config.json";

fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let device = Device::wgpu(Default::default());

    let config: BertConfig = serde_json::from_str(&fs::read_to_string(CONFIG)?)?;
    let mut model = BertForSequenceClassificationConfig::new().init(&config, &device);

    let start = Instant::now();
    let mut store = SafetensorsStore::from_file(WEIGHTS)
        .with_key_remapping(r"\.attention\.self\.", ".attention.self_attention.")
        .with_key_remapping(r"LayerNorm", "layer_norm")
        .with_from_adapter(PyTorchToBurnAdapter);

    let report = model.load_from(&mut store)?;
    println!("{report}");
    println!("loaded in {} ms", start.elapsed().as_millis());

    let mut tokenizer = Tokenizer::from_file(TOKENIZER)?;
    tokenizer.with_padding(Some(PaddingParams::default()));

    // Sanity check against the numbers published in the Qdrant blog post:
    // expected 0.8841 and 0.0035.
    let query = "rust programming";
    let documents = ["cargo builds rust code", "iron rusts in wet air"];

    let start = Instant::now();
    let scores = score(&model, &tokenizer, &device, query, &documents)?;
    println!(
        "\nscored {} docs in {} ms",
        documents.len(),
        start.elapsed().as_millis()
    );

    println!("\nquery: {query}");
    for (document, score) in documents.iter().zip(scores) {
        println!("  {score:.4}  {document}");
    }
    Ok(())
}

fn score(
    model: &BertForSequenceClassification,
    tokenizer: &Tokenizer,
    device: &Device,
    query: &str,
    documents: &[&str],
) -> Result<Vec<f32>, Box<dyn Error + Send + Sync>> {
    let pairs: Vec<(&str, &str)> = documents.iter().map(|d| (query, *d)).collect();
    let encodings = tokenizer.encode_batch(pairs, true)?;

    let batch = encodings.len();
    let seq_len = encodings[0].get_ids().len();
    let (mut ids, mut type_ids, mut mask) = (Vec::new(), Vec::new(), Vec::new());
    for encoding in &encodings {
        assert_eq!(
            encoding.get_ids().len(),
            seq_len,
            "padding did not equalise lengths"
        );
        ids.extend(encoding.get_ids().iter().map(|&v| v as i64));
        type_ids.extend(encoding.get_type_ids().iter().map(|&v| v as i64));
        mask.extend(encoding.get_attention_mask().iter().map(|&v| v as i64));
    }

    let input_ids = Tensor::<2, Int>::from_data(TensorData::new(ids, [batch, seq_len]), device);
    let token_type_ids =
        Tensor::<2, Int>::from_data(TensorData::new(type_ids, [batch, seq_len]), device);
    let attention_mask =
        Tensor::<2, Int>::from_data(TensorData::new(mask, [batch, seq_len]), device).equal_elem(0);

    let logits = model.forward(input_ids, attention_mask, token_type_ids); // [B, 1]
    Ok(sigmoid(logits.squeeze::<1>())
        .into_data()
        .try_to_vec::<f32>()
        .unwrap())
}
