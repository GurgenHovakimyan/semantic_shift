"""
Extract embeddings from processed IMDb data using RoBERTa.

This script generates sentence embeddings using the first token (index 0)
from the roberta-base model for both reference and stream sets.

Note: RoBERTa uses <s> as the start token (equivalent to [CLS] in BERT),
and the first token of the last hidden state is the standard representation
for classification tasks.
"""

import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel


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
    print("Loading RoBERTa model and tokenizer...")
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaModel.from_pretrained('roberta-base')
    model = model.to(device)
    model.eval()
    print("Model loaded successfully!")
    return tokenizer, model


def get_roberta_embeddings(
    df: pd.DataFrame,
    tokenizer: RobertaTokenizer,
    model: RobertaModel,
    device: torch.device,
    batch_size: int = 32
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract first token embeddings from the full_text column using RoBERTa.
    
    Args:
        df: DataFrame with 'full_text' and 'label' columns
        tokenizer: RoBERTa tokenizer
        model: RoBERTa model
        device: torch device
        batch_size: Batch size for processing
        
    Returns:
        Tuple of (embeddings tensor, labels tensor)
        - embeddings: (N, 768) tensor
        - labels: (N,) tensor
        
    Note:
        RoBERTa uses <s> as the start token. The first token (index 0) of
        the last hidden state is used as the sentence representation,
        similar to how [CLS] is used in BERT.
    """
    texts = df['full_text'].tolist()
    labels = df['label'].tolist()
    
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
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    return embeddings, labels_tensor


def main():
    # Define paths
    base_path = Path(__file__).parent.parent
    ref_input_path = base_path / 'data' / 'processed' / 'reference_set.pkl'
    stream_input_path = base_path / 'data' / 'processed' / 'stream_set.pkl'
    ref_output_path = base_path / 'data' / 'processed' / 'ref_embeddings_roberta.pt'
    stream_output_path = base_path / 'data' / 'processed' / 'stream_embeddings_roberta.pt'
    
    # Setup device and model
    device = get_device()
    tokenizer, model = load_model(device)
    
    # Process Reference Set
    print("\n" + "=" * 60)
    print("Processing REFERENCE SET with RoBERTa")
    print("=" * 60)
    
    df_ref = pd.read_pickle(ref_input_path)
    print(f"Loaded {len(df_ref)} samples from reference set")
    
    ref_embeddings, ref_labels = get_roberta_embeddings(df_ref, tokenizer, model, device)
    ref_dates = pd.to_datetime(df_ref['review_date']).values
    
    # Save reference embeddings
    ref_data = {
        'embeddings': ref_embeddings,
        'labels': ref_labels,
        'dates': ref_dates
    }
    torch.save(ref_data, ref_output_path)
    print(f"Saved reference embeddings to: {ref_output_path}")
    
    # Process Stream Set
    print("\n" + "=" * 60)
    print("Processing STREAM SET with RoBERTa")
    print("=" * 60)
    
    df_stream = pd.read_pickle(stream_input_path)
    print(f"Loaded {len(df_stream)} samples from stream set")
    
    stream_embeddings, stream_labels = get_roberta_embeddings(df_stream, tokenizer, model, device)
    stream_dates = pd.to_datetime(df_stream['review_date']).values
    
    # Save stream embeddings
    stream_data = {
        'embeddings': stream_embeddings,
        'labels': stream_labels,
        'dates': stream_dates
    }
    torch.save(stream_data, stream_output_path)
    print(f"Saved stream embeddings to: {stream_output_path}")
    
    # Verification
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    print("\n📊 REFERENCE EMBEDDINGS (RoBERTa):")
    print(f"   Embeddings shape: {ref_embeddings.shape}")
    print(f"   Labels shape: {ref_labels.shape}")
    print(f"   Expected: ({len(df_ref)}, 768)")
    print(f"   ✅ Match: {ref_embeddings.shape == (len(df_ref), 768)}")
    
    print("\n📊 STREAM EMBEDDINGS (RoBERTa):")
    print(f"   Embeddings shape: {stream_embeddings.shape}")
    print(f"   Labels shape: {stream_labels.shape}")
    print(f"   Expected: ({len(df_stream)}, 768)")
    print(f"   ✅ Match: {stream_embeddings.shape == (len(df_stream), 768)}")


if __name__ == '__main__':
    main()
