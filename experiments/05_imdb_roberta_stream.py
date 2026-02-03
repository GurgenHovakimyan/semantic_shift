"""
IMDb Stream Drift Detection Experiment using RoBERTa Embeddings.

This script analyzes semantic drift in IMDb reviews using MMD
by comparing sliding windows from the stream set against the reference set.
Uses RoBERTa embeddings instead of DistilBERT.

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
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.mmd import mmd_rbf_chunked as mmd_rbf


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
    data = torch.load(filepath, map_location='cpu', weights_only=False)
    # Keep on CPU for chunked MMD computation
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
    print("IMDb STREAM DRIFT DETECTION (RoBERTa)")
    print("=" * 60)
    print(f"Window Size: {WINDOW_SIZE}")
    print(f"Step Size: {STEP_SIZE}")
    
    # Setup paths
    base_path = project_root
    ref_path = base_path / 'data' / 'processed' / 'ref_embeddings_roberta.pt'
    stream_path = base_path / 'data' / 'processed' / 'stream_embeddings_roberta.pt'
    results_dir = base_path / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    output_csv = results_dir / 'imdb_roberta_drift.csv'
    output_plot = results_dir / 'roberta_drift_plot.png'
    
    # Setup device
    device = get_device()
    
    # Load embeddings
    print("\nLoading RoBERTa embeddings...")
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
    MAX_REF_SAMPLES = 500
    torch.manual_seed(42)
    
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
    
    if ref_embeddings.shape[0] > MAX_REF_SAMPLES:
        idx = torch.randperm(ref_embeddings.shape[0])[:MAX_REF_SAMPLES]
        ref_all_sample = ref_embeddings[idx]
    else:
        ref_all_sample = ref_embeddings
    
    print(f"Using {ref_all_sample.shape[0]} reference samples for global MMD")
    print(f"Using {ref_pos_sample.shape[0]} ref_pos, {ref_neg_sample.shape[0]} ref_neg for conditional MMD")
    
    # Stream Loop
    print("\nProcessing stream windows...")
    results = []
    n_samples = stream_embeddings.shape[0]
    n_windows = (n_samples - WINDOW_SIZE) // STEP_SIZE + 1
    
    # Subsample size for window
    WINDOW_SAMPLE = 500
    
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
                win_pos_sample = win_pos[idx]
            else:
                win_pos_sample = win_pos
                
            if win_neg.shape[0] > WINDOW_SAMPLE:
                idx = torch.randperm(win_neg.shape[0])[:WINDOW_SAMPLE]
                win_neg_sample = win_neg[idx]
            else:
                win_neg_sample = win_neg
            
            # Handle cases where window might not have both classes
            if win_pos_sample.shape[0] > 0 and win_neg_sample.shape[0] > 0:
                mmd_pos = mmd_rbf(ref_pos_sample, win_pos_sample).item()
                mmd_neg = mmd_rbf(ref_neg_sample, win_neg_sample).item()
                cond_mmd = 0.5 * mmd_pos + 0.5 * mmd_neg
            elif win_pos_sample.shape[0] > 0:
                cond_mmd = mmd_rbf(ref_pos_sample, win_pos_sample).item()
            elif win_neg_sample.shape[0] > 0:
                cond_mmd = mmd_rbf(ref_neg_sample, win_neg_sample).item()
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
    df_results.to_csv(output_csv, index=False)
    
    print(f"\n✅ Results saved to: {output_csv}")
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
    
    # Visualization
    print("\n" + "=" * 60)
    print("GENERATING PLOT")
    print("=" * 60)
    
    # Smoothing
    df_results['Global_Smooth'] = df_results['Global_MMD'].rolling(window=50, center=True).mean()
    df_results['Cond_Smooth'] = df_results['Conditional_MMD'].rolling(window=50, center=True).mean()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot Global MMD (Blue, Dashed)
    ax.plot(
        df_results['Date'], 
        df_results['Global_Smooth'], 
        color='blue', 
        linestyle='--', 
        linewidth=2,
        label='Global MMD (Marginal)'
    )
    
    # Plot Conditional MMD (Red, Solid)
    ax.plot(
        df_results['Date'], 
        df_results['Cond_Smooth'], 
        color='red', 
        linestyle='-', 
        linewidth=2,
        label='Conditional MMD (Semantic)'
    )
    
    # Title and labels
    ax.set_title('Drift Detection: RoBERTa Embeddings (IMDb)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('MMD Score', fontsize=12)
    
    # Grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=11)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45)
    
    # Tight layout
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_plot, dpi=150, bbox_inches='tight')
    print(f"✅ Plot saved to: {output_plot}")


if __name__ == '__main__':
    main()
