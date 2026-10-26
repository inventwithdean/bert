use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear},
    tensor::{Bool, Device, Int, module::attention, ops::AttentionModuleOptions, s},
};

use crate::config::BertConfig;

#[derive(Module, Debug)]
pub struct BertEmbeddings {
    word_embeddings: Embedding,
    position_embeddings: Embedding,
    token_type_embeddings: Embedding,
    LayerNorm: LayerNorm,
    position_ids: Tensor<2, Int>,
}

impl BertEmbeddings {
    pub fn forward(&self, input_ids: Tensor<2, Int>, token_type_ids: Tensor<2, Int>) -> Tensor<3> {
        let [_batch_size, seq_len] = input_ids.dims();

        let position_ids = self.position_ids.clone().slice(s![.., 0..seq_len]);
        let inputs_embeds = self.word_embeddings.forward(input_ids);
        let token_type_embeddings = self.token_type_embeddings.forward(token_type_ids);
        let embeddings = inputs_embeds + token_type_embeddings;

        let position_embeddings = self.position_embeddings.forward(position_ids);
        let embeddings = embeddings + position_embeddings;
        self.LayerNorm.forward(embeddings)
    }
}

#[derive(Config, Debug)]
pub struct BertEmbeddingsConfig {}

impl BertEmbeddingsConfig {
    pub fn init(&self, config: &BertConfig, device: &Device) -> BertEmbeddings {
        let word_embeddings =
            EmbeddingConfig::new(config.vocab_size, config.hidden_size).init(device);
        let position_embeddings =
            EmbeddingConfig::new(config.max_position_embeddings, config.hidden_size).init(device);
        let token_type_embeddings =
            EmbeddingConfig::new(config.type_vocab_size, config.hidden_size).init(device);
        let position_ids =
            Tensor::arange(0..config.max_position_embeddings as i64, device).unsqueeze_dim::<2>(0);
        let layer_norm = LayerNormConfig::new(config.hidden_size)
            .with_epsilon(config.layer_norm_eps)
            .init(device);
        BertEmbeddings {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            LayerNorm: layer_norm,
            position_ids,
        }
    }
}

#[derive(Module, Debug)]
pub struct BertSelfAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    num_attention_heads: usize,
    attention_head_size: usize,
}

impl BertSelfAttention {
    pub fn forward(
        &self,
        hidden_states: Tensor<3>,
        attention_mask: Option<Tensor<4, Bool>>,
    ) -> Tensor<3> {
        let [batch_size, seq_len, _hidden_size] = hidden_states.dims();
        let hidden_shape = [
            batch_size,
            seq_len,
            self.num_attention_heads,
            self.attention_head_size,
        ];

        // -> (B, seq_len, all_head_size)
        // -> (B, seq_len, num_attention_heads, attention_head_size)
        // -> (B, num_attention_heads, seq_len, attention_head_size)

        let query_layer = self
            .query
            .forward(hidden_states.clone())
            .reshape(hidden_shape)
            .swap_dims(1, 2);
        let key_layer = self
            .key
            .forward(hidden_states.clone())
            .reshape(hidden_shape)
            .swap_dims(1, 2);
        let value_layer = self
            .value
            .forward(hidden_states)
            .reshape(hidden_shape)
            .swap_dims(1, 2);

        let output = attention(
            query_layer,
            key_layer,
            value_layer,
            attention_mask,
            None,
            AttentionModuleOptions::default(),
        ); // (B, num_attention_heads, seq_len, attention_head_size)

        let output = output.swap_dims(1, 2); // (B, seq_len, num_attention_heads, attention_head_size)
        output.reshape([
            batch_size,
            seq_len,
            self.num_attention_heads * self.attention_head_size,
        ])
    }
}
