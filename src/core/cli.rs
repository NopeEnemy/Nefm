use burn::{
    prelude::*,
    record::{CompactRecorder, Recorder},
    tensor::backend::AutodiffBackend,
};

use crate::core::{
    data::{dataset::*, tokenizer::*},
    inference::{InferInput, infer_loop},
    train::*,
};

use clap::{Args, Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(name = "nefm")]
#[command(author = "Artem Moiseev <ar.m.moiseev@gmail.com>")]
#[command(version = "0.1.0")]
#[command(about = "NEFM: Expiremental language model on Burn framework", long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Run model inference (text generation)
    Run {
        /// Path to the model artifacts directory
        artifact_dir: String,
        /// Sampling temperature (e.g., 0.7)
        #[arg(short, long, default_value_t = 0.7)]
        temperature: f32,
        /// Top-K tokens filtering limit
        #[arg(short, long, default_value_t = 50)]
        top_k: usize,
        /// Input prompt text for generation
        prompt: String,
    },
    /// Start model training
    Train(TrainArgs),
    /// Display the total number of parameters for a trained model
    Params {
        /// Path to the model artifacts directory
        artifact_dir: String,
    },
    /// Train a BPE tokenizer on the dataset
    Tokenizer {
        /// Output path to save the trained tokenizer
        artifact_dir: String,
        /// Target vocabulary size
        target_size: usize,
    },
}

#[derive(Args, Debug)]
pub struct TrainArgs {
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

pub fn run<B: Backend, AB: AutodiffBackend>(
    device: &B::Device,
    a_device: &AB::Device,
    cli: &Cli,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    match &cli.command {
        Commands::Run {
            artifact_dir,
            temperature,
            top_k,
            prompt,
        } => {
            let input = InferInput::<B>::from_artifact_dir(
                device,
                artifact_dir,
                temperature.to_owned(),
                top_k.to_owned(),
                prompt.to_owned(),
            );
            let _ = infer_loop::<B>(device, input);
        }

        Commands::Train(args) => {
            parse_args_and_train::<AB>(a_device, args)?;
        }

        Commands::Params { artifact_dir } => {
            let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
                .map_err(|_| "Конфигурация модели не найдена. Сначала запустите обучение.")?;

            let record = CompactRecorder::new()
                .load(format!("{artifact_dir}/model").into(), device)
                .map_err(|_| "Обученная модель не найдена. Сначала запустите обучение.")?;

            let model = config.model.init::<B>(device).load_record(record);
            println!("Model params: {}", model.num_params());
        }

        Commands::Tokenizer {
            artifact_dir,
            target_size,
        } => {
            let dataset_train = ModelDataset::train("dataset")?;
            let dataset_valid = ModelDataset::valid("dataset")?;

            let tokenizer = Tokenizer::train_from_db(
                &dataset_train,
                &dataset_valid,
                target_size.to_owned(),
                true,
            );
            tokenizer.save(artifact_dir)?;
        }
    }

    Ok(())
}
