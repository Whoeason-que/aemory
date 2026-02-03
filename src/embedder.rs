use anyhow::Result;
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

pub struct Embedder {
    model: TextEmbedding,
}

impl Embedder {
    pub fn new(model_name: &str) -> Result<Self> {
        let mut options = InitOptions::new(EmbeddingModel::BGEBaseENV15);
        options.show_download_progress = true;

        if model_name.contains("multilingual") {
            options.model_name = EmbeddingModel::MultilingualE5Small; // A safe guess for multilingual?
        }

        let model = TextEmbedding::try_new(options)?;
        Ok(Self { model })
    }

    pub fn generate_embeddings(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let embeddings = self.model.embed(texts.to_vec(), None)?;
        Ok(embeddings)
    }
}
