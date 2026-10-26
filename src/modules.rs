use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig},
    tensor::{
        Bool, Device, Int, activation::gelu, module::attention, ops::AttentionModuleOptions, s,
    },
};

use crate::config::BertConfig;

#[derive(Module, Debug)]
pub struct BertEmbeddings {
    word_embeddings: Embedding,
    position_embeddings: Embedding,
    token_type_embeddings: Embedding,
    layer_norm: LayerNorm,
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
        self.layer_norm.forward(embeddings)
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
            layer_norm,
            position_ids,
        }
    }
}

#[derive(Module, Debug)]
pub struct BertSelfOutput {
    dense: Linear,
    layer_norm: LayerNorm,
}

impl BertSelfOutput {
    pub fn forward(&self, hidden_states: Tensor<3>, input_tensor: Tensor<3>) -> Tensor<3> {
        let hidden_states = self.dense.forward(hidden_states);
        self.layer_norm.forward(hidden_states + input_tensor)
    }
}

#[derive(Config, Debug)]
pub struct BertSelfOutputConfig {}

impl BertSelfOutputConfig {
    pub fn init(&self, config: &BertConfig, device: &Device) -> BertSelfOutput {
        let dense = LinearConfig::new(config.hidden_size, config.hidden_size).init(device);
        let layer_norm = LayerNormConfig::new(config.hidden_size)
            .with_epsilon(config.layer_norm_eps)
            .init(device);
        BertSelfOutput { dense, layer_norm }
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
    pub fn forward(&self, hidden_states: Tensor<3>, attention_mask: Tensor<4, Bool>) -> Tensor<3> {
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
            Some(attention_mask),
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

#[derive(Config, Debug)]
pub struct BertSelfAttentionConfig {}

impl BertSelfAttentionConfig {
    pub fn init(&self, config: &BertConfig, device: &Device) -> BertSelfAttention {
        let num_attention_heads = config.num_attention_heads;
        let attention_head_size = config.hidden_size / config.num_attention_heads;
        let all_head_size = num_attention_heads * attention_head_size;
        let query = LinearConfig::new(config.hidden_size, all_head_size).init(device);
        let key = LinearConfig::new(config.hidden_size, all_head_size).init(device);
        let value = LinearConfig::new(config.hidden_size, all_head_size).init(device);
        BertSelfAttention {
            query,
            key,
            value,
            num_attention_heads,
            attention_head_size,
        }
    }
}

#[derive(Module, Debug)]
pub struct BertAttention {
    self_attention: BertSelfAttention,
    output: BertSelfOutput,
}

impl BertAttention {
    pub fn forward(&self, hidden_states: Tensor<3>, attention_mask: Tensor<4, Bool>) -> Tensor<3> {
        let attention_output = self
            .self_attention
            .forward(hidden_states.clone(), attention_mask);
        self.output.forward(attention_output, hidden_states)
    }
}

#[derive(Config, Debug)]
pub struct BertAttentionConfig {}

impl BertAttentionConfig {
    pub fn init(&self, config: &BertConfig, device: &Device) -> BertAttention {
        let self_attention = BertSelfAttentionConfig::new().init(config, device);
        let output = BertSelfOutputConfig::new().init(config, device);
        BertAttention {
            self_attention,
            output,
        }
    }
}

#[derive(Module, Debug)]
pub struct BertIntermediate {
    dense: Linear,
}

impl BertIntermediate {
    pub fn forward(&self, hidden_states: Tensor<3>) -> Tensor<3> {
        let hidden_states = self.dense.forward(hidden_states);
        gelu(hidden_states)
    }
}

#[derive(Config, Debug)]
pub struct BertIntermediateConfig {}

impl BertIntermediateConfig {
    pub fn init(&self, config: &BertConfig, device: &Device) -> BertIntermediate {
        let dense = LinearConfig::new(config.hidden_size, config.intermediate_size).init(device);

        BertIntermediate { dense }
    }
}

#[derive(Module, Debug)]
pub struct BertOutput {
    dense: Linear,
    layer_norm: LayerNorm,
}

impl BertOutput {
    pub fn forward(&self, hidden_states: Tensor<3>, input_tensor: Tensor<3>) -> Tensor<3> {
        let hidden_states = self.dense.forward(hidden_states);
        self.layer_norm.forward(hidden_states + input_tensor)
    }
}

#[derive(Config, Debug)]
pub struct BertOutputConfig {}

impl BertOutputConfig {
    pub fn init(&self, config: &BertConfig, device: &Device) -> BertOutput {
        let dense = LinearConfig::new(config.intermediate_size, config.hidden_size).init(device);
        let layer_norm = LayerNormConfig::new(config.hidden_size)
            .with_epsilon(config.layer_norm_eps)
            .init(device);
        BertOutput { dense, layer_norm }
    }
}

#[derive(Module, Debug)]
pub struct BertLayer {
    attention: BertAttention,
    intermediate: BertIntermediate,
    output: BertOutput,
}

impl BertLayer {
    pub fn forward(&self, hidden_states: Tensor<3>, attention_mask: Tensor<4, Bool>) -> Tensor<3> {
        let attention_output = self.attention.forward(hidden_states, attention_mask);
        let intermediate_output = self.intermediate.forward(attention_output.clone());
        self.output.forward(intermediate_output, attention_output)
    }
}

#[derive(Config, Debug)]
pub struct BertLayerConfig {}

impl BertLayerConfig {
    pub fn init(&self, config: &BertConfig, device: &Device) -> BertLayer {
        let attention = BertAttentionConfig::new().init(config, device);
        let intermediate = BertIntermediateConfig::new().init(config, device);
        let output = BertOutputConfig::new().init(config, device);
        BertLayer {
            attention,
            intermediate,
            output,
        }
    }
}
