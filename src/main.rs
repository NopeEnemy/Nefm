// main.rs
use burn::{
    backend::{Autodiff, Wgpu},
    prelude::*,
    tensor::bf16,
};
use clap::Parser;
use nefm::core::cli::{Cli, run};
type B = Wgpu;
type AB = Autodiff<Wgpu<bf16, i32, u32>>;

fn main() {
    let device = Default::default();
    let _tensor = Tensor::<B, 2>::from_data([[2.0]], &device);

    let args = Cli::parse();

    let result = run::<B, AB>(&device, &device, &args);

    println!("{:?}", result);
}
