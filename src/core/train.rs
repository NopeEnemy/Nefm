// train.rs

use std::fs::{create_dir_all, remove_dir_all};

use burn::{
    data::{
        dataloader::{DataLoader, DataLoaderBuilder},
        dataset::Dataset,
    },
    grad_clipping::GradientClippingConfig,
    lr_scheduler::cosine::CosineAnnealingLrSchedulerConfig,
    optim::AdamConfig,
    prelude::*,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{
        ClassificationOutput, Learner, SupervisedTraining, TrainOutput, TrainStep,
        metric::{LossMetric, PerplexityMetric},
    },
};

use crate::core::{
    cli::TrainArgs,
    data::{batcher::*, dataset::*, tokenizer::*},
    model::*,
};
use std::sync::Arc;

#[derive(Debug, Config)]
pub struct TrainingConfig {
    pub model: ModelConfig,
    pub optim: AdamConfig,
    pub tokenizer: Tokenizer,
    pub dataset_dir: String,

    #[config(default = 10)]
    pub n_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 512)]
    pub max_seq_length: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-4)]
    pub lr: f64,
    pub min_lr: f64,
}

fn create_artifact_dir(dir: &str) {
    remove_dir_all(dir).ok();
    create_dir_all(dir).ok();
}

pub fn train<B: AutodiffBackend>(
    device: &B::Device,
    config: TrainingConfig,
    artifact_dir: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let model = config.model.init::<B>(device);
    config.save(format!("{artifact_dir}/config.json"))?;

    B::seed(device, config.seed);

    let dataset_train = ModelDataset::train("dataset")?;
    let dataset_valid = ModelDataset::valid("dataset")?;

    let batcher = ModelBatcher {
        tokenizer: config.tokenizer,
        max_seq_lenght: config.max_seq_length,
    };

    let n_iters = (dataset_train.len() + config.batch_size - 1) / config.batch_size;
    let n_iters = n_iters * config.n_epochs;

    let dataloader_train: Arc<dyn DataLoader<B, ModelBatch<B>>> =
        DataLoaderBuilder::new(batcher.clone())
            .batch_size(config.batch_size)
            .shuffle(config.seed)
            .num_workers(config.num_workers)
            .build(dataset_train);

    let dataloader_valid = DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(dataset_valid);

    let training = SupervisedTraining::new(artifact_dir, dataloader_train, dataloader_valid)
        .num_epochs(config.n_epochs)
        .metrics((PerplexityMetric::new(), LossMetric::new()));

    let lr = CosineAnnealingLrSchedulerConfig::new(config.lr, n_iters)
        .with_min_lr(config.min_lr)
        .init()?;

    let learner = Learner::new(model, config.optim.init(), lr);

    let result = training.launch(learner);

    result
        .model
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())?;

    Ok(())
}

impl<B: AutodiffBackend> TrainStep for Model<B> {
    type Input = ModelBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, item: Self::Input) -> burn::train::TrainOutput<Self::Output> {
        let item = self.forward_training(item);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

#[derive(Debug)]
pub struct TrainInput {
    pub artifact_dir: String,
    pub n_epochs: usize,
    pub n_layer: usize,
    pub n_heads: usize,
    pub d_model: usize,
    pub d_hidden: usize,
    pub context_len: usize,
    pub batch_size: usize,
    pub n_workers: usize,
    pub seed: u64,
    pub lr: f64,
    pub min_lr: f64,
    pub tokenizer_path: String,
}

pub fn parse_args_and_train<B: AutodiffBackend>(
    device: &B::Device,
    train_input: &TrainArgs,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    create_artifact_dir(&train_input.artifact_dir);

    let tokenizer = Tokenizer::load(&train_input.tokenizer_path);

    let config = TrainingConfig {
        model: ModelConfig::new(
            train_input.context_len,
            tokenizer.len(),
            train_input.d_model,
            train_input.n_layer,
            train_input.n_heads,
            train_input.d_hidden,
        ),
        optim: AdamConfig::new().with_grad_clipping(Some(GradientClippingConfig::Norm(1.0))),
        tokenizer,
        dataset_dir: "dataset".to_string(),
        n_epochs: train_input.n_epochs,
        batch_size: train_input.batch_size,
        num_workers: train_input.n_workers,
        max_seq_length: train_input.context_len,
        seed: train_input.seed,
        lr: train_input.lr,
        min_lr: train_input.min_lr,
    };

    train::<B>(device, config, &train_input.artifact_dir)
}
