"""
IMDb Stream Drift Detection Experiment.

This script analyzes semantic drift in IMDb reviews using MMD
by comparing sliding windows from the stream set against the reference set.

Metrics:
- Global Drift: MMD between all reference and all window samples
- Semantic Drift: Conditional MMD averaged across sentiment classes
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from src.mmd import mmd_rbf


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


def load_embeddings(filepath: Path, device: torch.device) -> dict:
    """Load embeddings and move to device."""
    data = torch.load(filepath, map_location=device, weights_only=False)
    data['embeddings'] = data['embeddings'].to(device)
    data['labels'] = data['labels'].to(device)
    return data


def split_by_label(embeddings: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split embeddings into positive (1) and negative (0) subsets."""
    pos_mask = labels == 1
    neg_mask = labels == 0
    return embeddings[pos_mask], embeddings[neg_mask]


def main():
    # Configuration
    WINDOW_SIZE = 1000  # Reviews per batch
    STEP_SIZE = 500     # Overlapping windows
    
    print("=" * 60)
    print("IMDb STREAM DRIFT DETECTION EXPERIMENT")
    print("=" * 60)
    print(f"Window Size: {WINDOW_SIZE}")
    print(f"Step Size: {STEP_SIZE}")
    
    # Setup paths
    base_path = project_root
    ref_path = base_path / 'data' / 'processed' / 'ref_embeddings.pt'
    stream_path = base_path / 'data' / 'processed' / 'stream_embeddings.pt'
    results_dir = base_path / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / 'imdb_drift_results.csv'
    
    # Setup device
    device = get_device()
    
    # Load embeddings
    print("\nLoading embeddings...")
    ref_data = load_embeddings(ref_path, device)
    stream_data = load_embeddings(stream_path, device)
    
    ref_embeddings = ref_data['embeddings']
    ref_labels = ref_data['labels']
    stream_embeddings = stream_data['embeddings']
    stream_labels = stream_data['labels']
    stream_dates = stream_data['dates']
    
    print(f"Reference: {ref_embeddings.shape[0]} samples")
    print(f"Stream: {stream_embeddings.shape[0]} samples")
    
    # Reference Preparation: Split by label
    print("\nPreparing reference subsets...")
    ref_pos, ref_neg = split_by_label(ref_embeddings, ref_labels)
    print(f"Reference Positive: {ref_pos.shape[0]} samples")
    print(f"Reference Negative: {ref_neg.shape[0]} samples")
    
    # Sample reference for efficiency
    # Note: Using small samples to avoid memory issues with kernel matrix
    MAX_REF_SAMPLES = 500
    WINDOW_SAMPLE = 500  # Subsample window for MMD computation
    
    if ref_pos.shape[0] > MAX_REF_SAMPLES:
        idx = torch.randperm(ref_pos.shape[0])[:MAX_REF_SAMPLES]
        ref_pos_sample = ref_pos[idx]
    else:
        ref_pos_sample = ref_pos
        
    if ref_neg.shape[0] > MAX_REF_SAMPLES:
        idx = torch.randperm(ref_neg.shape[0])[:MAX_REF_SAMPLES]
        ref_neg_sample = ref_neg[idx]
    else:
        ref_neg_sample = ref_neg
    
    # Sample all reference for global MMD
    if ref_embeddings.shape[0] > MAX_REF_SAMPLES:
        idx = torch.randperm(ref_embeddings.shape[0])[:MAX_REF_SAMPLES]
        ref_all_sample = ref_embeddings[idx]
    else:
        ref_all_sample = ref_embeddings
    
    print(f"Using {ref_all_sample.shape[0]} reference samples for global MMD")
    print(f"Using {ref_pos_sample.shape[0]} ref_pos, {ref_neg_sample.shape[0]} ref_neg for conditional MMD")
    
    # Move reference samples to CPU for MMD computation (avoid GPU memory issues)
    ref_all_sample = ref_all_sample.cpu()
    ref_pos_sample = ref_pos_sample.cpu()
    ref_neg_sample = ref_neg_sample.cpu()
    stream_embeddings = stream_embeddings.cpu()
    stream_labels = stream_labels.cpu()
    print("Moved tensors to CPU for MMD computation")
    
    # Stream Loop
    print("\nProcessing stream windows...")
    results = []
    n_samples = stream_embeddings.shape[0]
    n_windows = (n_samples - WINDOW_SIZE) // STEP_SIZE + 1
    
    with torch.no_grad():
        for i in tqdm(range(0, n_samples - WINDOW_SIZE + 1, STEP_SIZE), total=n_windows, desc="Windows"):
            # Get window
            window_emb = stream_embeddings[i:i + WINDOW_SIZE]
            window_labels = stream_labels[i:i + WINDOW_SIZE]
            window_dates = stream_dates[i:i + WINDOW_SIZE]
            
            # Timestamp: Average date of reviews in window
            avg_date = pd.to_datetime(window_dates).mean()
            
            # Subsample window for MMD computation
            if window_emb.shape[0] > WINDOW_SAMPLE:
                idx = torch.randperm(window_emb.shape[0])[:WINDOW_SAMPLE]
                window_emb_sample = window_emb[idx]
            else:
                window_emb_sample = window_emb
            
            # Metric 1: Global Drift (Reference_All vs Window_All)
            global_mmd = mmd_rbf(ref_all_sample, window_emb_sample).item()
            
            # Metric 2: Semantic Drift (Conditional MMD)
            win_pos, win_neg = split_by_label(window_emb, window_labels)
            
            # Subsample class-specific windows
            if win_pos.shape[0] > WINDOW_SAMPLE:
                idx = torch.randperm(win_pos.shape[0])[:WINDOW_SAMPLE]
                win_pos = win_pos[idx]
            if win_neg.shape[0] > WINDOW_SAMPLE:
                idx = torch.randperm(win_neg.shape[0])[:WINDOW_SAMPLE]
                win_neg = win_neg[idx]
            
            # Handle cases where window might not have both classes
            if win_pos.shape[0] > 0 and win_neg.shape[0] > 0:
                mmd_pos = mmd_rbf(ref_pos_sample, win_pos).item()
                mmd_neg = mmd_rbf(ref_neg_sample, win_neg).item()
                cond_mmd = 0.5 * mmd_pos + 0.5 * mmd_neg
            elif win_pos.shape[0] > 0:
                cond_mmd = mmd_rbf(ref_pos_sample, win_pos).item()
            elif win_neg.shape[0] > 0:
                cond_mmd = mmd_rbf(ref_neg_sample, win_neg).item()
            else:
                cond_mmd = float('nan')
            
            results.append({
                'Date': avg_date,
                'Global_MMD': global_mmd,
                'Conditional_MMD': cond_mmd,
                'Window_Start': i,
                'Window_End': i + WINDOW_SIZE,
                'Pos_Count': win_pos.shape[0],
                'Neg_Count': win_neg.shape[0]
            })
    
    # Convert to DataFrame and save
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('Date').reset_index(drop=True)
    df_results.to_csv(output_path, index=False)
    
    print(f"\n✅ Results saved to: {output_path}")
    print(f"Total windows processed: {len(df_results)}")
    
    # Verification: Print first 5 rows
    print("\n" + "=" * 60)
    print("FIRST 5 ROWS OF RESULTS")
    print("=" * 60)
    print(df_results.head().to_string(index=False))
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Date Range: {df_results['Date'].min()} to {df_results['Date'].max()}")
    print(f"Global MMD  - Mean: {df_results['Global_MMD'].mean():.6f}, Std: {df_results['Global_MMD'].std():.6f}")
    print(f"Cond. MMD   - Mean: {df_results['Conditional_MMD'].mean():.6f}, Std: {df_results['Conditional_MMD'].std():.6f}")


if __name__ == '__main__':
    main()
