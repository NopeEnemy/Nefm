use crate::core::data::dataset::*;
use burn::{data::dataloader::Dataset, prelude::*};
use serde::{Deserialize, Serialize, ser::StdError};
use tokenizers::tokenizer::{
    DecoderWrapper, NormalizerWrapper, PostProcessorWrapper, PreTokenizerWrapper, TokenizerImpl,
};
use tokenizers::{
    AddedToken,
    models::bpe::{BPE, BpeTrainerBuilder},
};

type BpeTokenizer = TokenizerImpl<
    BPE,
    NormalizerWrapper,
    PreTokenizerWrapper,
    PostProcessorWrapper,
    DecoderWrapper,
>;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Tokenizer {
    pub inner: BpeTokenizer,
}

impl Tokenizer {
    pub fn train(str: String, target_vocab_size: usize) -> Self {
        let bpe = BPE::default();
        let mut hf_tokenizer = BpeTokenizer::new(bpe);

        let mut trainer = BpeTrainerBuilder::default()
            .vocab_size(target_vocab_size)
            .build();

        hf_tokenizer
            .train(&mut trainer, str.chars().map(|c| c.to_string()))
            .expect("Error while training tokenizer");

        Self {
            inner: hf_tokenizer,
        }
    }

    pub fn train_from_db(
        dataset_train: &ModelDataset,
        dataset_valid: &ModelDataset,
        target_vocab_size: usize,
        special: bool,
    ) -> Self {
        let bpe = BPE::default();

        let mut special_tokens = Vec::new();
        if special {
            special_tokens = vec![
                AddedToken::from("[PAD]", true),
                AddedToken::from("[START]", true),
                AddedToken::from("[END]", true),
                AddedToken::from("[UNK]", true),
            ];
        }

        let mut hf_tokenizer = BpeTokenizer::new(bpe);

        let mut trainer = BpeTrainerBuilder::default()
            .show_progress(true)
            .vocab_size(target_vocab_size)
            .special_tokens(special_tokens)
            .build();

        let train_iter =
            (0..dataset_train.len()).filter_map(|i| dataset_train.get(i).map(|item| item.content));

        let valid_iter =
            (0..dataset_valid.len()).filter_map(|i| dataset_valid.get(i).map(|item| item.content));

        let combined_iterator = train_iter.chain(valid_iter);

        hf_tokenizer
            .train(&mut trainer, combined_iterator)
            .expect("Error while training tokenizer");

        Self {
            inner: hf_tokenizer,
        }
    }

    pub fn load(path: &str) -> Self {
        let inner = BpeTokenizer::from_file(path).expect("Не удалось загрузить JSON токенизатора");
        Self { inner }
    }

    pub fn save(&self, path: &str) -> Result<(), Box<dyn StdError + Send + Sync + 'static>> {
        self.inner.save(path, true)?;

        Ok(())
    }

    pub fn encode(&self, str: &str, special: bool) -> Vec<usize> {
        let encoding = self
            .inner
            .encode(str, special)
            .expect("Ошибка при кодировании текста");

        encoding.get_ids().iter().map(|&id| id as usize).collect()
    }

    pub fn len(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    pub fn is_empty(&self) -> bool {
        self.inner.get_vocab_size(true) > 0
    }

    pub fn decode(&self, input: &Vec<i32>) -> String {
        let ids: Vec<u32> = input.iter().map(|n| n.to_owned().to_u32()).collect();
        self.inner.decode(&ids, false).unwrap()
    }
    pub fn decode_simple(&self, input: usize) -> String {
        self.inner.decode(&[input.to_u32()], false).unwrap()
    }

    pub fn decode_from_tensor<B: Backend>(&self, input: Tensor<B, 2, Int>) -> String {
        let input: Vec<i32> = input.to_data().to_vec().unwrap();
        self.decode(&input)
    }
}
