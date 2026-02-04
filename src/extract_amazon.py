"""
Extract embeddings from Amazon Electronics reviews using DistilBERT.

This script generates sentence embeddings using the [CLS] token
from the distilbert-base-uncased model for the balanced Amazon dataset.
"""

import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import DistilBertTokenizer, DistilBertModel

# Configuration
INPUT_PATH = "data/processed/amazon_electronics_balanced.pkl"
OUTPUT_PATH = "data/processed/amazon_embeddings.pt"
MODEL_NAME = "distilbert-base-uncased"
BATCH_SIZE = 32


def get_device() -> torch.device:
    """Detect and return the best available device."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS (Apple Silicon)")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def load_model(device: torch.device) -> tuple[DistilBertTokenizer, DistilBertModel]:
    """Load DistilBERT tokenizer and model."""
    print(f"Loading {MODEL_NAME} model and tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    model = DistilBertModel.from_pretrained(MODEL_NAME)
    model = model.to(device)
    model.eval()
    print("Model loaded successfully!")
    return tokenizer, model


def extract_embeddings(
    df: pd.DataFrame,
    tokenizer: DistilBertTokenizer,
    model: DistilBertModel,
    device: torch.device,
    batch_size: int = 32
) -> torch.Tensor:
    """
    Extract [CLS] token embeddings from the text column.
    
    Args:
        df: DataFrame with 'text' column
        tokenizer: DistilBERT tokenizer
        model: DistilBERT model
        device: torch device
        batch_size: Batch size for processing
        
    Returns:
        embeddings: (N, 768) tensor
    """
    texts = df['text'].tolist()
    
    all_embeddings = []
    
    # Process in batches
    num_batches = (len(texts) + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), total=num_batches, desc="Extracting embeddings"):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            
            # Move to device
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Get [CLS] token embedding (index 0)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            
            # Move to CPU to save memory
            all_embeddings.append(cls_embeddings.cpu())
    
    # Concatenate all batches
    embeddings = torch.cat(all_embeddings, dim=0)
    
    return embeddings


def main():
    """Main execution function."""
    # Setup device
    device = get_device()
    
    # Load model
    tokenizer, model = load_model(device)
    
    # Load data
    print(f"\nLoading data from {INPUT_PATH}...")
    df = pd.read_pickle(INPUT_PATH)
    print(f"Loaded {len(df)} samples.")
    
    # Extract embeddings
    print(f"\nExtracting embeddings with batch size {BATCH_SIZE}...")
    embeddings = extract_embeddings(df, tokenizer, model, device, batch_size=BATCH_SIZE)
    
    # Prepare labels and dates
    labels = torch.tensor(df['label'].values, dtype=torch.long)
    dates = df['review_date'].values  # Keep as numpy array for flexibility
    
    # Save output
    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_dict = {
        'embeddings': embeddings,
        'labels': labels,
        'dates': dates
    }
    
    torch.save(output_dict, output_path)
    print(f"\nSaved embeddings to {OUTPUT_PATH}")
    
    # Verification
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Dates shape: {dates.shape}")
    print(f"Expected shape: [10000, 768]")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
