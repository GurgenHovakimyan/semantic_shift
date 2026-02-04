"""
Extract embeddings from Amazon Electronics reviews using RoBERTa.

This script generates sentence embeddings using the first token (index 0)
from the roberta-base model for the balanced Amazon dataset.

Note: RoBERTa uses <s> as the start token (equivalent to [CLS] in BERT),
and the first token of the last hidden state is the standard representation
for classification tasks.
"""

import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel

# Configuration
INPUT_PATH = "data/processed/amazon_electronics_balanced.pkl"
OUTPUT_PATH = "data/processed/amazon_roberta_embeddings.pt"
MODEL_NAME = "roberta-base"
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


def load_model(device: torch.device) -> tuple[RobertaTokenizer, RobertaModel]:
    """Load RoBERTa tokenizer and model."""
    print(f"Loading {MODEL_NAME} model and tokenizer...")
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    model = RobertaModel.from_pretrained(MODEL_NAME)
    model = model.to(device)
    model.eval()
    print("Model loaded successfully!")
    return tokenizer, model


def extract_embeddings(
    df: pd.DataFrame,
    tokenizer: RobertaTokenizer,
    model: RobertaModel,
    device: torch.device,
    batch_size: int = 32
) -> torch.Tensor:
    """
    Extract first token embeddings from the text column using RoBERTa.
    
    Args:
        df: DataFrame with 'text' column
        tokenizer: RoBERTa tokenizer
        model: RoBERTa model
        device: torch device
        batch_size: Batch size for processing
        
    Returns:
        embeddings: (N, 768) tensor
        
    Note:
        RoBERTa uses <s> as the start token. The first token (index 0) of
        the last hidden state is used as the sentence representation,
        similar to how [CLS] is used in BERT.
    """
    texts = df['text'].tolist()
    
    all_embeddings = []
    
    # Process in batches
    num_batches = (len(texts) + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), total=num_batches, desc="Extracting embeddings"):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize with RoBERTa-specific settings
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
            
            # Get first token embedding (index 0) - equivalent to <s> token
            # This is the standard representation for classification in RoBERTa
            first_token_embeddings = outputs.last_hidden_state[:, 0, :]
            
            # Move to CPU to save memory
            all_embeddings.append(first_token_embeddings.cpu())
    
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
