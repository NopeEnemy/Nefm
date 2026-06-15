// lib.rs

pub mod core {
    pub mod data {
        pub mod batcher;
        pub mod dataset;
        pub mod tokenizer;
    }
    pub mod cli;
    pub mod inference;
    pub mod model;
    pub mod train;
}
