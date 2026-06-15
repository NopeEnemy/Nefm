// batcher.rs

use crate::core::data::{dataset::*, tokenizer::*};
use burn::{data::dataloader::batcher::Batcher, nn::attention::generate_padding_mask, prelude::*};

#[derive(Debug, Clone)]
pub struct ModelBatch<B: Backend> {
    pub tokens: Tensor<B, 2, Int>,
    pub mask: Tensor<B, 2, Bool>,
    pub targets: Tensor<B, 2, Int>,
}

#[derive(Debug, Clone)]
pub struct ModelBatcher {
    pub tokenizer: Tokenizer,
    pub max_seq_lenght: usize,
}

impl<B: Backend> Batcher<B, SQLItem, ModelBatch<B>> for ModelBatcher {
    fn batch(&self, items: Vec<SQLItem>, device: &Device<B>) -> ModelBatch<B> {
        let mut tokens_list = Vec::with_capacity(items.len());

        for item in items {
            let t = self.tokenizer.encode(&item.content, true);

            tokens_list.push(t);
        }

        let mask = generate_padding_mask::<B>(0, tokens_list, Some(self.max_seq_lenght), device);

        let [batch_size, seq_length] = mask.tensor.clone().dims();

        let tokens = mask
            .tensor
            .clone()
            .slice([0..batch_size, 0..seq_length - 1]);

        let targets = mask.tensor.clone().slice([0..batch_size, 1..seq_length]);

        let mask = mask.mask.slice([0..batch_size, 0..seq_length - 1]);

        ModelBatch {
            tokens,
            mask,
            targets,
        }
    }
}
