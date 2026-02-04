"""
Amazon Electronics RoBERTa Stream Drift Detection Experiment.

Cross-domain validation using Amazon Electronics reviews with RoBERTa embeddings.
Analyzes semantic drift using MMD by comparing sliding windows against a reference set.

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
from src.mmd import mmd_rbf_chunked


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


def plot_drift_results(df_results: pd.DataFrame, output_path: Path) -> None:
    """Create and save the drift detection plot."""
    # Smoothing: Rolling mean with window for visualization
    smooth_window = min(5, len(df_results) // 3)  # Adaptive smoothing window
    if smooth_window >= 3:
        df_results['Global_Smooth'] = df_results['Global_MMD'].rolling(window=smooth_window, center=True).mean()
        df_results['Cond_Smooth'] = df_results['Conditional_MMD'].rolling(window=smooth_window, center=True).mean()
    else:
        df_results['Global_Smooth'] = df_results['Global_MMD']
        df_results['Cond_Smooth'] = df_results['Conditional_MMD']
    
    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot Global MMD (Marginal) - Blue, Dashed
    ax.plot(
        df_results['Date'], 
        df_results['Global_Smooth'], 
        color='blue', 
        linestyle='--', 
        linewidth=2,
        label='Global MMD (Marginal)'
    )
    
    # Plot Conditional MMD (Semantic) - Red, Solid
    ax.plot(
        df_results['Date'], 
        df_results['Cond_Smooth'], 
        color='red', 
        linestyle='-', 
        linewidth=2,
        label='Conditional MMD (Semantic)'
    )
    
    # Title and labels
    ax.set_title('Cross-Domain Validation: Amazon Electronics (RoBERTa)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('MMD Score', fontsize=12)
    
    # Grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=11)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Tight layout to prevent label cutoff
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")


def main():
    # Configuration
    WINDOW_SIZE = 1000  # Reviews per window
    STEP_SIZE = 500     # Overlapping windows
    REF_SIZE = 2000     # Reference set size (oldest reviews)
    
    print("=" * 60)
    print("AMAZON ELECTRONICS ROBERTA DRIFT DETECTION EXPERIMENT")
    print("=" * 60)
    print(f"Reference Size: {REF_SIZE} (oldest reviews)")
    print(f"Stream Size: 8000 (remaining reviews)")
    print(f"Window Size: {WINDOW_SIZE}")
    print(f"Step Size: {STEP_SIZE}")
    
    # Setup paths
    base_path = project_root
    embeddings_path = base_path / 'data' / 'processed' / 'amazon_roberta_embeddings.pt'
    results_dir = base_path / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_output_path = results_dir / 'amazon_roberta_drift_results.csv'
    plot_output_path = results_dir / 'amazon_roberta_drift_plot.png'
    
    # Setup device
    device = get_device()
    
    # Load embeddings
    print("\nLoading RoBERTa embeddings...")
    data = load_embeddings(embeddings_path, device)
    
    all_embeddings = data['embeddings']
    all_labels = data['labels']
    all_dates = data['dates']
    
    print(f"Total samples: {all_embeddings.shape[0]}")
    
    # Split into Reference and Stream
    print("\nSplitting data...")
    ref_embeddings = all_embeddings[:REF_SIZE]
    ref_labels = all_labels[:REF_SIZE]
    
    stream_embeddings = all_embeddings[REF_SIZE:]
    stream_labels = all_labels[REF_SIZE:]
    stream_dates = all_dates[REF_SIZE:]
    
    print(f"Reference: {ref_embeddings.shape[0]} samples (oldest)")
    print(f"Stream: {stream_embeddings.shape[0]} samples (newer)")
    
    # Reference Preparation: Split by label
    print("\nPreparing reference subsets...")
    ref_pos, ref_neg = split_by_label(ref_embeddings, ref_labels)
    print(f"Reference Positive: {ref_pos.shape[0]} samples")
    print(f"Reference Negative: {ref_neg.shape[0]} samples")
    
    # Sample reference for efficiency
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
    
    # Move tensors to CPU for MMD computation (avoid GPU memory issues)
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
            global_mmd = mmd_rbf_chunked(ref_all_sample, window_emb_sample, chunk_size=128).item()
            
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
                mmd_pos = mmd_rbf_chunked(ref_pos_sample, win_pos, chunk_size=128).item()
                mmd_neg = mmd_rbf_chunked(ref_neg_sample, win_neg, chunk_size=128).item()
                cond_mmd = 0.5 * mmd_pos + 0.5 * mmd_neg
            elif win_pos.shape[0] > 0:
                cond_mmd = mmd_rbf_chunked(ref_pos_sample, win_pos, chunk_size=128).item()
            elif win_neg.shape[0] > 0:
                cond_mmd = mmd_rbf_chunked(ref_neg_sample, win_neg, chunk_size=128).item()
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
    df_results.to_csv(csv_output_path, index=False)
    
    print(f"\n✅ Results saved to: {csv_output_path}")
    print(f"Total windows processed: {len(df_results)}")
    
    # Create and save plot
    plot_drift_results(df_results, plot_output_path)
    
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
