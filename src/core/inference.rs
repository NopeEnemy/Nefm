// inference.rs

use rand::{
    distr::{Distribution, weighted::WeightedIndex},
    rng,
};

use std::{
    error::Error,
    f32,
    io::{self, Write},
};

use crate::core::{data::tokenizer::Tokenizer, model::Model, train::*};
use burn::{
    nn::attention::generate_padding_mask,
    prelude::*,
    record::{CompactRecorder, Recorder},
};

pub struct InferInput<B: Backend> {
    pub model: Model<B>,
    pub tokenizer: Tokenizer,
    pub prompt: String,
    pub temperature: f32,
    pub top_k: usize,
    pub contex_len: usize,
}

impl<B: Backend> InferInput<B> {
    pub fn from_artifact_dir(
        device: &B::Device,
        artifact_dir: &str,
        temperature: f32,
        top_k: usize,
        prompt: String,
    ) -> Self {
        let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
            .expect("Config should be exist for the model; run train first");

        let record = CompactRecorder::new()
            .load(format!("{artifact_dir}/model").into(), device)
            .expect("Trained model should exist; run train first");

        let model = config.model.init::<B>(device).load_record(record);
        let tokenizer = config.tokenizer;

        Self {
            model,
            tokenizer,
            prompt,
            temperature,
            top_k,
            contex_len: config.max_seq_length,
        }
    }
}

pub fn infer<B: Backend>(
    device: &B::Device,
    input: Vec<usize>,
    model: &mut Model<B>,
    temperature: f32,
    top_k: usize,
) -> usize {
    let pad_id = 0;
    let current_len = input.len();

    let mask = generate_padding_mask(pad_id, vec![input], Some(current_len), device);

    let output = model.forward_infer(mask.tensor, mask.mask);
    let [batch_size, seq_len, _vocab_size] = output.dims();

    let last_token_logits = output.slice([0..batch_size, (seq_len - 1)..seq_len]);
    let mut logits: Vec<f32> = last_token_logits.to_data().to_vec().unwrap();

    if temperature > 0.0 {
        for logit in logits.iter_mut() {
            *logit /= temperature;
        }
    }

    let mut indexed_logits: Vec<(usize, f32)> = logits.into_iter().enumerate().collect();

    indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    if top_k > 0 && top_k < indexed_logits.len() {
        for item in indexed_logits.iter_mut().skip(top_k) {
            item.1 = f32::NEG_INFINITY;
        }
    }

    let max_logit = indexed_logits
        .iter()
        .map(|&(_, l)| l)
        .fold(f32::NEG_INFINITY, f32::max);

    let mut weights = Vec::with_capacity(indexed_logits.len());
    let mut indices = Vec::with_capacity(indexed_logits.len());

    for (idx, logit) in indexed_logits {
        if logit == f32::NEG_INFINITY {
            continue;
        }
        let exp = (logit - max_logit).exp();
        weights.push(exp);
        indices.push(idx);
    }

    let mut rng = rng();
    let dist = WeightedIndex::new(&weights).expect("Ошибка создания распределения вероятностей");
    let sampled_array_idx = dist.sample(&mut rng);

    indices[sampled_array_idx]
}

pub fn infer_loop<B: Backend>(
    device: &B::Device,
    infer_input: InferInput<B>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let mut model = infer_input.model;

    let tokenizer = infer_input.tokenizer;

    let mut input = tokenizer.encode(&infer_input.prompt, true);
    let mut current_input = input.clone();

    loop {
        if input.len() >= infer_input.contex_len {
            println!("\n[Достигнут максимальный размер контекста]");
            break Ok(());
        }

        let output = infer(
            device,
            current_input.clone(),
            &mut model,
            infer_input.temperature,
            infer_input.top_k,
        );

        let pred = tokenizer.decode_simple(output);

        io::stdout().flush().unwrap();
        print!("{pred}");
        if pred == "[END]" {
            break Ok(());
        }

        input.push(output);

        current_input = vec![output];
    }
}
