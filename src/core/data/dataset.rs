// dataset.rs

use burn::data::dataset::{Dataset, SqliteDataset, SqliteDatasetError};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SQLItem {
    pub content: String,
}

pub struct ModelDataset {
    dataset: SqliteDataset<SQLItem>,
}

impl Dataset<SQLItem> for ModelDataset {
    fn get(&self, index: usize) -> Option<SQLItem> {
        self.dataset.get(index)
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

impl ModelDataset {
    pub fn train(path: &str) -> Result<Self, SqliteDatasetError> {
        Self::new(path, "dataset.db", "train")
    }

    pub fn valid(path: &str) -> Result<Self, SqliteDatasetError> {
        Self::new(path, "dataset.db", "valid")
    }

    pub fn test(path: &str) -> Result<Self, SqliteDatasetError> {
        Self::new(path, "dataset.db", "test")
    }

    pub fn new(path: &str, file_name: &str, table_name: &str) -> Result<Self, SqliteDatasetError> {
        let dataset = SqliteDataset::from_db_file(format!("{path}/{file_name}"), table_name)?;

        Ok(Self { dataset })
    }
}
