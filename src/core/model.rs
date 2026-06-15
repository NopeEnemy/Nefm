// model.rs

use burn::{
    nn::{
        Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig,
        RotaryEncoding, RotaryEncodingConfig,
        attention::generate_autoregressive_mask,
        loss::CrossEntropyLossConfig,
        transformer::{PositionWiseFeedForward, PositionWiseFeedForwardConfig},
    },
    prelude::*,
    tensor::activation::softmax,
    train::{ClassificationOutput, InferenceStep},
};

use crate::core::data::batcher::ModelBatch;

#[derive(Debug, Config)]
pub struct ModelConfig {
    context_length: usize,
    vocab_size: usize,
    d_model: usize,
    n_layers: usize,
    n_heads: usize,
    d_hidden: usize,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        let token_embedding = EmbeddingConfig::new(self.vocab_size, self.d_model).init::<B>(device);

        let blocks: Vec<Block<B>> = (0..self.n_layers)
            .map(|_| {
                BlockConfig::new(
                    self.d_model,
                    self.d_hidden,
                    self.n_heads,
                    self.n_layers,
                    self.context_length,
                )
                .init(device)
            })
            .collect();

        let lr = LinearConfig::new(self.d_model, self.vocab_size).init::<B>(device);

        let final_norm = LayerNormConfig::new(self.d_model).init(device);

        Model {
            token_embedding,
            blocks,
            lr,
            final_norm,
        }
    }
}

#[derive(Debug, Module)]
pub struct Model<B: Backend> {
    pub token_embedding: Embedding<B>,
    blocks: Vec<Block<B>>,
    pub lr: Linear<B>,
    pub final_norm: LayerNorm<B>,
}

impl<B: Backend> Model<B> {
    pub fn forward(&self, input: Tensor<B, 2, Int>, pad_mask: Tensor<B, 2, Bool>) -> Tensor<B, 3> {
        let x = self.token_embedding.forward(input.clone());

        let x = self
            .blocks
            .iter()
            .fold(x, |x, block| block.forward(x, pad_mask.clone()));

        let x = self.final_norm.forward(x);

        self.lr.forward(x)
    }

    pub fn forward_infer(
        &mut self,
        input: Tensor<B, 2, Int>,
        pad_mask: Tensor<B, 2, Bool>,
    ) -> Tensor<B, 3> {
        let x = self.token_embedding.forward(input.clone());

        let x = self
            .blocks
            .iter_mut()
            .fold(x, |x, block| block.forward_infer(x, pad_mask.clone()));

        let x = self.final_norm.forward(x);

        self.lr.forward(x)
    }

    pub fn forward_training(&self, item: ModelBatch<B>) -> ClassificationOutput<B> {
        let output = self.forward(item.tokens.clone(), item.mask.clone());
        let device = item.mask.device();

        let [batch_size, seq_lenght, vocab_size] = output.dims();

        let output_flatten = output
            .clone()
            .reshape([batch_size * seq_lenght, vocab_size]);

        let targets_squeezed = item.targets.clone().reshape([seq_lenght * batch_size]);

        let loss = CrossEntropyLossConfig::new()
            .init::<B>(&device)
            .forward(output_flatten.clone(), targets_squeezed.clone());

        ClassificationOutput {
            loss,
            output: output_flatten,
            targets: targets_squeezed,
        }
    }
}

#[derive(Debug, Module)]
struct Block<B: Backend> {
    norm1: LayerNorm<B>,
    attn: Attention<B>,
    norm2: LayerNorm<B>,
    ffn: PositionWiseFeedForward<B>,
}

impl<B: Backend> Block<B> {
    fn forward(&self, input: Tensor<B, 3>, mask_pad: Tensor<B, 2, Bool>) -> Tensor<B, 3> {
        let x = self.norm1.forward(input.clone());
        let x = self.attn.forward(x, mask_pad);
        let x = input + x;

        let residual = x.clone();
        let normalized_x = self.norm2.forward(x);

        residual + self.ffn.forward(normalized_x)
    }

    fn forward_infer(&mut self, input: Tensor<B, 3>, mask_pad: Tensor<B, 2, Bool>) -> Tensor<B, 3> {
        let x = self.norm1.forward(input.clone());
        let x = self.attn.forward_cache(x, mask_pad);
        let x = input + x;

        let residual = x.clone();
        let normalized_x = self.norm2.forward(x);

        residual + self.ffn.forward(normalized_x)
    }
}

#[derive(Debug, Config)]
pub struct BlockConfig {
    d_model: usize,
    d_hidden: usize,
    n_heads: usize,
    n_layers: usize,
    context_len: usize,
    #[config(default = 0.2)]
    dropout: f64,
}

impl BlockConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> Block<B> {
        let norm1 = LayerNormConfig::new(self.d_model).init::<B>(device);
        let attn = AttentionConfig::new(self.d_model, self.n_heads, self.context_len).init(device);
        let ffn = PositionWiseFeedForwardConfig::new(self.d_model, self.d_hidden)
            .with_dropout(self.dropout)
            .init(device);

        let norm2 = LayerNormConfig::new(self.d_model).init::<B>(device);

        Block {
            norm1,
            attn,
            ffn,
            norm2,
        }
    }
}

impl<B: Backend> InferenceStep for Model<B> {
    type Input = ModelBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, item: Self::Input) -> Self::Output {
        self.forward_training(item)
    }
}

#[derive(Debug, Module)]
struct Attention<B: Backend> {
    query: Linear<B>,
    key: Linear<B>,
    value: Linear<B>,
    output: Linear<B>,
    d_model: usize,
    n_heads: usize,
    rotary: RotaryEncoding<B>,
    k_cache: Option<Tensor<B, 4>>,
    v_cache: Option<Tensor<B, 4>>,
}

impl<B: Backend> Attention<B> {
    pub fn forward(&self, input: Tensor<B, 3>, padding_mask: Tensor<B, 2, Bool>) -> Tensor<B, 3> {
        let [batch_size, seq_len, _] = input.dims();

        let q = self.query.forward(input.clone());
        let k = self.key.forward(input.clone());
        let v = self.value.forward(input);

        let q = self.reshape_heads(q, batch_size, seq_len);
        let k = self.reshape_heads(k, batch_size, seq_len);
        let v = self.reshape_heads(v, batch_size, seq_len);

        let q = self.rotary.forward(q);
        let k = self.rotary.forward(k);

        let x = q.matmul(k.swap_dims(2, 3));
        let x = x.div_scalar((self.d_model / self.n_heads).isqrt().to_f32());

        let casual_mask = generate_autoregressive_mask::<B>(batch_size, seq_len, &x.device());
        let casual_mask = casual_mask.unsqueeze_dim(1);

        let pad_mask = padding_mask.unsqueeze_dim::<3>(1).unsqueeze_dim::<4>(2);

        let mask = casual_mask.bool_or(pad_mask);

        let x = x.mask_fill(mask, -1.0e4);
        let x = softmax(x, 3);

        let x = x.matmul(v);
        let x = x
            .swap_dims(1, 2)
            .reshape([batch_size, seq_len, self.d_model]);

        self.output.forward(x)
    }

    fn forward_cache(
        &mut self,
        input: Tensor<B, 3>,
        padding_mask: Tensor<B, 2, Bool>,
    ) -> Tensor<B, 3> {
        let [batch_size, seq_len, _] = input.dims();

        let q = self.query.forward(input.clone());

        let k = self.key.forward(input.clone());
        let v = self.value.forward(input);

        let q = self.reshape_heads(q, batch_size, seq_len);
        let k = self.reshape_heads(k, batch_size, seq_len);
        let v = self.reshape_heads(v, batch_size, seq_len);

        let (k, v) = self.update_cache(k, v);

        let q = self.rotary.forward(q);
        let k = self.rotary.forward(k);

        let x = q.matmul(k.swap_dims(2, 3));
        let x = x.div_scalar((self.d_model / self.n_heads).isqrt().to_f32());

        let x = if seq_len > 1 {
            let casual_mask = generate_autoregressive_mask::<B>(batch_size, seq_len, &x.device());
            let casual_mask = casual_mask.unsqueeze_dim(1);

            let pad_mask = padding_mask.unsqueeze_dim::<3>(1).unsqueeze_dim::<4>(2);

            let mask = casual_mask.bool_or(pad_mask);

            x.mask_fill(mask, -1.0e4)
        } else {
            x
        };

        let x = softmax(x, 3);

        let x = x.matmul(v);
        let x = x
            .swap_dims(1, 2)
            .reshape([batch_size, seq_len, self.d_model]);

        self.output.forward(x)
    }

    fn update_cache(
        &mut self,
        k_new: Tensor<B, 4>,
        v_new: Tensor<B, 4>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        match (self.k_cache.take(), self.v_cache.take()) {
            (Some(k), Some(v)) => {
                let k = Tensor::cat(vec![k, k_new.clone()], 2);
                let v = Tensor::cat(vec![v, v_new.clone()], 2);

                self.k_cache = Some(k.clone());
                self.v_cache = Some(v.clone());

                (k, v)
            }
            _ => {
                self.k_cache = Some(k_new.clone());
                self.v_cache = Some(v_new.clone());
                (k_new, v_new)
            }
        }
    }

    fn reshape_heads(&self, x: Tensor<B, 3>, batch_size: usize, seq_len: usize) -> Tensor<B, 4> {
        x.reshape([
            batch_size,
            seq_len,
            self.n_heads,
            (self.d_model / self.n_heads),
        ])
        .swap_dims(1, 2)
    }
}

#[derive(Debug, Config)]
struct AttentionConfig {
    d_model: usize,
    n_heads: usize,
    context_len: usize,
}

impl AttentionConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> Attention<B> {
        let query = LinearConfig::new(self.d_model, self.d_model).init(device);
        let key = LinearConfig::new(self.d_model, self.d_model).init(device);
        let value = LinearConfig::new(self.d_model, self.d_model).init(device);

        let output = LinearConfig::new(self.d_model, self.d_model).init(device);

        let rotary =
            RotaryEncodingConfig::new(self.context_len, self.d_model / self.n_heads).init(device);

        Attention {
            query,
            key,
            value,
            output,
            d_model: self.d_model,
            n_heads: self.n_heads,
            rotary,
            k_cache: None,
            v_cache: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use burn::{
        backend::{Wgpu, wgpu::WgpuDevice},
        prelude::*,
    };

    type B = Wgpu;

    const DEVICE: Device<B> = WgpuDevice::DefaultDevice;

    use crate::core::{data::batcher::ModelBatch, model::ModelConfig};

    #[test]
    fn forward_training_test() {
        let model = ModelConfig::new(256, 128, 64, 3, 4, 64).init::<B>(&DEVICE);

        let item = ModelBatch {
            tokens: Tensor::<B, 2, Int>::from_data([[10, 15, 21]], &DEVICE),
            targets: Tensor::<B, 2, Int>::from_data([[0, 35, 6]], &DEVICE),
            mask: Tensor::<B, 2, Bool>::from_data([[true, false, false]], &DEVICE),
        };

        model.forward_training(item);
    }

    #[test]
    fn test_model_initialization() {
        let config = ModelConfig::new(128, 1000, 64, 2, 4, 128);
        let model = config.init::<B>(&DEVICE);

        assert!(model.num_params() > 0, "Модель не должна быть пустой");
    }

    #[test]
    fn test_forward_training_shapes() {
        let config = ModelConfig::new(32, 100, 64, 2, 4, 128);
        let model = config.init::<B>(&DEVICE);

        let tokens = Tensor::<B, 2, Int>::from_data([[1, 2, 3, 4], [5, 6, 7, 8]], &DEVICE);
        let targets = Tensor::<B, 2, Int>::from_data([[2, 3, 4, 0], [6, 7, 8, 0]], &DEVICE);
        let mask = Tensor::<B, 2, Bool>::from_data(
            [[false, false, false, false], [false, false, false, false]],
            &DEVICE,
        );

        let item = ModelBatch {
            tokens,
            targets,
            mask,
        };
        let output = model.forward_training(item);

        assert_eq!(output.loss.dims(), [1]);
    }
}
