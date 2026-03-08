use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

pub struct Embedder {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl Embedder {
    pub fn new(model_name: &str) -> Result<Self> {
        println!("📦 Loading model: {}", model_name);

        // Map common model names to HF repo IDs
        let repo_id = match model_name {
            "BAAI/bge-small-en-v1.5" => "BAAI/bge-small-en-v1.5",
            "BAAI/bge-base-en-v1.5" => "BAAI/bge-base-en-v1.5",
            "sentence-transformers/all-MiniLM-L6-v2" => "sentence-transformers/all-MiniLM-L6-v2",
            _ if model_name.contains("multilingual") => "intfloat/multilingual-e5-small",
            _ => model_name,
        };

        // Download model files from Hugging Face
        let api = Api::new().context("Failed to create HF API client")?;
        let repo = api.repo(Repo::new(repo_id.to_string(), RepoType::Model));
        
        let config_path = repo.get("config.json").context("Failed to download config.json")?;
        let tokenizer_path = repo.get("tokenizer.json").context("Failed to download tokenizer.json")?;
        let weights_path = repo.get("model.safetensors")
            .or_else(|_| repo.get("pytorch_model.bin"))
            .context("Failed to download model weights")?;

        // Load config
        let config = std::fs::read_to_string(config_path)
            .context("Failed to read config file")?;
        let config: Config = serde_json::from_str(&config)
            .context("Failed to parse config.json")?;

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        // Setup device (CPU for universal compatibility)
        let device = Device::Cpu;

        // Load model weights
        let vb = if weights_path.extension().and_then(|s| s.to_str()) == Some("safetensors") {
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DTYPE, &device)? }
        } else {
            VarBuilder::from_pth(&weights_path, DTYPE, &device)?
        };

        let model = BertModel::load(vb, &config)
            .context("Failed to load BERT model")?;

        println!("✅ Model loaded successfully");

        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    pub fn generate_embeddings(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut all_embeddings = Vec::with_capacity(texts.len());

        for text in texts {
            // Tokenize
            let encoding = self
                .tokenizer
                .encode(text.as_str(), true)
                .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

            let tokens = encoding.get_ids();
            let token_ids = Tensor::new(tokens, &self.device)?
                .unsqueeze(0)?; // Add batch dimension

            // Create attention mask (all 1s for valid tokens)
            let attention_mask = vec![1u32; tokens.len()];
            let attention_mask = Tensor::new(attention_mask.as_slice(), &self.device)?
                .unsqueeze(0)?;

            // Create token type IDs (all 0s for single sequence)
            let token_type_ids = vec![0u32; tokens.len()];
            let token_type_ids = Tensor::new(token_type_ids.as_slice(), &self.device)?
                .unsqueeze(0)?;

            // Forward pass
            let embeddings = self.model.forward(&token_ids, &token_type_ids, Some(&attention_mask))?;

            // Mean pooling: average over sequence length (dim=1)
            let (_batch_size, seq_len, _hidden_size) = embeddings.dims3()?;
            let pooled = (embeddings.sum(1)? / (seq_len as f64))?;

            // Extract as Vec<f32>
            let embedding_vec = pooled.squeeze(0)?.to_vec1::<f32>()?;

            all_embeddings.push(embedding_vec);
        }

        Ok(all_embeddings)
    }
}
