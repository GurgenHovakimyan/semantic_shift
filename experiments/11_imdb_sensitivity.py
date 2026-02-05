"""
IMDb Sensitivity Analysis - Minimum Sample Size for Drift Detection.

This experiment determines the minimum sample size (N) required to 
statistically detect semantic drift in IMDb reviews using Conditional MMD.

Methodology:
- Test various sample sizes: [100, 200, 500, 1000, 2000, 3000]
- Use "Concept Swap" method to simulate guaranteed drift signal
- Drifted stream: 50% label swap (maximum drift)
- Clean stream: 0% label swap (control/baseline)
- Repeat 5 times per sample size for statistical robustness
- Report Mean ± Std Dev of MMD scores

Output:
- Table of results (N, Mean MMD, Std Dev)
- Line plot with error bars saved to results/imdb_sensitivity_plot.png
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.mmd import mmd_rbf_chunked as mmd_rbf


def load_embeddings(filepath: Path) -> dict:
    """Load embeddings from file."""
    data = torch.load(filepath, map_location='cpu', weights_only=False)
    return data


def split_by_label(embeddings: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split embeddings into positive (1) and negative (0) subsets."""
    pos_mask = labels == 1
    neg_mask = labels == 0
    return embeddings[pos_mask], embeddings[neg_mask]


def flip_labels(labels: torch.Tensor, flip_ratio: float, seed: int = None) -> torch.Tensor:
    """
    Randomly flip a percentage of labels.
    
    Args:
        labels: Original labels (0 or 1)
        flip_ratio: Percentage of labels to flip (0.0 to 1.0)
        seed: Random seed for reproducibility
        
    Returns:
        Corrupted labels with flip_ratio% flipped
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    corrupted = labels.clone()
    n_samples = len(labels)
    n_flip = int(n_samples * flip_ratio)
    
    if n_flip > 0:
        # Randomly select indices to flip
        flip_indices = torch.randperm(n_samples)[:n_flip]
        # Flip: 0 -> 1, 1 -> 0
        corrupted[flip_indices] = 1 - corrupted[flip_indices]
    
    return corrupted


def compute_conditional_mmd(X_ref: torch.Tensor, Y_ref: torch.Tensor,
                            X_stream: torch.Tensor, Y_stream: torch.Tensor,
                            max_sample_size: int = 500) -> float:
    """
    Compute Conditional MMD between reference and stream distributions.
    
    Conditional MMD = 0.5 * MMD(P(E|Y=1), Q(E|Y=1)) + 0.5 * MMD(P(E|Y=0), Q(E|Y=0))
    
    Args:
        X_ref: Reference embeddings
        Y_ref: Reference labels
        X_stream: Stream embeddings
        Y_stream: Stream labels (potentially corrupted)
        max_sample_size: Maximum subsample size for each class (for efficiency)
        
    Returns:
        Conditional MMD score
    """
    # Split by labels
    ref_pos, ref_neg = split_by_label(X_ref, Y_ref)
    stream_pos, stream_neg = split_by_label(X_stream, Y_stream)
    
    # Subsample for efficiency (cap at max_sample_size)
    if ref_pos.shape[0] > max_sample_size:
        ref_pos = ref_pos[torch.randperm(ref_pos.shape[0])[:max_sample_size]]
    if ref_neg.shape[0] > max_sample_size:
        ref_neg = ref_neg[torch.randperm(ref_neg.shape[0])[:max_sample_size]]
    if stream_pos.shape[0] > max_sample_size:
        stream_pos = stream_pos[torch.randperm(stream_pos.shape[0])[:max_sample_size]]
    if stream_neg.shape[0] > max_sample_size:
        stream_neg = stream_neg[torch.randperm(stream_neg.shape[0])[:max_sample_size]]
    
    # Compute class-conditional MMD
    mmd_pos = mmd_rbf(ref_pos, stream_pos, chunk_size=128).item()
    mmd_neg = mmd_rbf(ref_neg, stream_neg, chunk_size=128).item()
    
    # Average conditional MMD
    cond_mmd = 0.5 * mmd_pos + 0.5 * mmd_neg
    
    return cond_mmd


def run_sensitivity_trial(all_embeddings: torch.Tensor, all_labels: torch.Tensor,
                          sample_size: int, swap_ratio: float, seed: int) -> float:
    """
    Run a single sensitivity trial.
    
    Args:
        all_embeddings: Full dataset embeddings
        all_labels: Full dataset labels
        sample_size: Number of samples for reference and stream
        swap_ratio: Label swap ratio (0.0 = clean, 0.5 = max drift)
        seed: Random seed for reproducibility
        
    Returns:
        Conditional MMD score
    """
    torch.manual_seed(seed)
    
    total_samples = all_embeddings.shape[0]
    
    # Sample 2*N points and split into reference and stream
    all_indices = torch.randperm(total_samples)[:sample_size * 2]
    ref_indices = all_indices[:sample_size]
    stream_indices = all_indices[sample_size:]
    
    X_ref = all_embeddings[ref_indices]
    Y_ref = all_labels[ref_indices]
    
    X_stream = all_embeddings[stream_indices]
    Y_stream = all_labels[stream_indices]
    
    # Apply concept swap to stream labels
    Y_stream_corrupted = flip_labels(Y_stream, swap_ratio, seed=seed + 1000)
    
    # Compute Conditional MMD
    cond_mmd = compute_conditional_mmd(X_ref, Y_ref, X_stream, Y_stream_corrupted)
    
    return cond_mmd


def main():
    print("=" * 70)
    print("IMDb SENSITIVITY ANALYSIS: Minimum Sample Size for Drift Detection")
    print("=" * 70)
    
    # Setup paths
    base_path = project_root
    embeddings_path = base_path / 'data' / 'processed' / 'ref_embeddings.pt'
    output_path = base_path / 'results' / 'imdb_sensitivity_plot.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load IMDb embeddings (DistilBERT)
    print("\nLoading IMDb embeddings (DistilBERT)...")
    data = load_embeddings(embeddings_path)
    all_embeddings = data['embeddings']
    all_labels = data['labels']
    
    print(f"Total samples available: {all_embeddings.shape[0]}")
    print(f"Embedding dimension: {all_embeddings.shape[1]}")
    
    # Check label distribution
    n_pos = (all_labels == 1).sum().item()
    n_neg = (all_labels == 0).sum().item()
    print(f"Label distribution: {n_pos} positive, {n_neg} negative")
    
    # Experimental setup
    N_values = [100, 200, 500, 1000, 2000, 3000]
    SWAP_RATIO_DRIFTED = 0.5  # 50% swap (max drift)
    SWAP_RATIO_CLEAN = 0.0    # 0% swap (no drift)
    NUM_REPEATS = 5
    
    print(f"\nExperimental Setup:")
    print(f"  Sample sizes (N): {N_values}")
    print(f"  Drifted stream swap rate: {SWAP_RATIO_DRIFTED * 100:.0f}%")
    print(f"  Clean stream swap rate: {SWAP_RATIO_CLEAN * 100:.0f}%")
    print(f"  Repeats per N: {NUM_REPEATS}")
    
    # Results storage
    results_drifted = {N: [] for N in N_values}
    results_clean = {N: [] for N in N_values}
    
    print("\n" + "-" * 70)
    print("Running experiments...")
    print("-" * 70)
    
    with torch.no_grad():
        for N in N_values:
            print(f"\nSample Size N = {N}")
            
            for rep in range(NUM_REPEATS):
                seed = 42 + rep * 100
                
                # Drifted stream (50% swap)
                mmd_drifted = run_sensitivity_trial(
                    all_embeddings, all_labels, N, SWAP_RATIO_DRIFTED, seed
                )
                results_drifted[N].append(mmd_drifted)
                
                # Clean stream (0% swap)
                mmd_clean = run_sensitivity_trial(
                    all_embeddings, all_labels, N, SWAP_RATIO_CLEAN, seed
                )
                results_clean[N].append(mmd_clean)
                
                print(f"  Rep {rep + 1}: Drifted MMD = {mmd_drifted:.6f}, Clean MMD = {mmd_clean:.6f}")
    
    # Compute statistics
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    stats_drifted = {}
    stats_clean = {}
    
    print("\n{:^10} | {:^20} | {:^20}".format("N", "Drifted (50% swap)", "Clean (0% swap)"))
    print("-" * 55)
    
    for N in N_values:
        mean_drifted = np.mean(results_drifted[N])
        std_drifted = np.std(results_drifted[N])
        mean_clean = np.mean(results_clean[N])
        std_clean = np.std(results_clean[N])
        
        stats_drifted[N] = {'mean': mean_drifted, 'std': std_drifted}
        stats_clean[N] = {'mean': mean_clean, 'std': std_clean}
        
        print(f"{N:^10} | {mean_drifted:.6f} ± {std_drifted:.6f} | {mean_clean:.6f} ± {std_clean:.6f}")
    
    # Additional table: Just the main results
    print("\n" + "-" * 50)
    print("Main Results Table (Drifted Stream - 50% Swap)")
    print("-" * 50)
    print("{:>10} | {:>12} | {:>12}".format("N", "Mean MMD", "Std Dev"))
    print("-" * 40)
    
    for N in N_values:
        mean_mmd = stats_drifted[N]['mean']
        std_mmd = stats_drifted[N]['std']
        print(f"{N:>10} | {mean_mmd:>12.6f} | {std_mmd:>12.6f}")
    
    # Plotting
    print("\n" + "=" * 70)
    print("Generating plot...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract data for plotting
    means_drifted = [stats_drifted[N]['mean'] for N in N_values]
    stds_drifted = [stats_drifted[N]['std'] for N in N_values]
    means_clean = [stats_clean[N]['mean'] for N in N_values]
    stds_clean = [stats_clean[N]['std'] for N in N_values]
    
    # Plot Drifted stream (50% swap) - Red with error bars
    ax.errorbar(
        N_values, 
        means_drifted, 
        yerr=stds_drifted,
        color='red', 
        linestyle='-', 
        linewidth=2.5,
        marker='o',
        markersize=10,
        capsize=5,
        capthick=2,
        label='Drifted (50% Swap)'
    )
    
    # Plot Clean stream (0% swap) - Blue with error bars
    ax.errorbar(
        N_values, 
        means_clean, 
        yerr=stds_clean,
        color='blue', 
        linestyle='--', 
        linewidth=2.5,
        marker='s',
        markersize=10,
        capsize=5,
        capthick=2,
        label='Clean (0% Swap)'
    )
    
    # Formatting
    ax.set_xlabel('Sample Size (N)', fontsize=14, fontweight='bold')
    ax.set_ylabel('MMD Score', fontsize=14, fontweight='bold')
    ax.set_title('IMDb: Detection Sensitivity vs. Sample Size', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Set x-axis to show all N values
    ax.set_xticks(N_values)
    ax.set_xticklabels([str(n) for n in N_values])
    
    # Ensure y-axis starts at 0
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    
    plt.close()  # Close figure to free memory
    
    # Statistical interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    # Find minimum N where drifted MMD is significantly above clean
    print("\nDetection Reliability Analysis:")
    for N in N_values:
        mean_d = stats_drifted[N]['mean']
        std_d = stats_drifted[N]['std']
        mean_c = stats_clean[N]['mean']
        std_c = stats_clean[N]['std']
        
        # Rough separation metric: (mean_drifted - mean_clean) / (std_drifted + std_clean)
        if (std_d + std_c) > 0:
            separation = (mean_d - mean_c) / (std_d + std_c)
        else:
            separation = float('inf')
        
        status = "✓ DETECTABLE" if separation > 1.0 else "? MARGINAL" if separation > 0.5 else "✗ UNRELIABLE"
        print(f"  N = {N:4}: Separation = {separation:.2f} → {status}")
    
    # Compare with Amazon results
    print("\n" + "-" * 50)
    print("Cross-Dataset Comparison (IMDb vs Amazon)")
    print("-" * 50)
    print("Expected: Similar separation patterns across datasets")
    print("The 'Golden Ratio' from concept swap experiments suggests")
    print("IMDb and Amazon have ~0.99x similar sensitivity.")
    
    print("\n" + "=" * 70)
    print("IMDb Sensitivity Analysis Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
