use serde::Deserialize;

#[derive(Deserialize)]
pub struct BertConfig {
    pub hidden_size: usize,
    pub initializer_range: f32,
    pub intermediate_size: usize,
    pub layer_norm_eps: f64,
    pub max_position_embeddings: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub pad_token_id: usize,
    pub type_vocab_size: usize,
    pub vocab_size: usize,
}
