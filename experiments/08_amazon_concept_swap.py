"""
Amazon Concept Swap Experiment - Cross-Domain Validation of Semantic Drift Detection.

This experiment validates that Conditional MMD can detect semantic drift
on the Amazon Electronics dataset, even when Global MMD is perfectly stable.

Key Insight: By keeping embeddings identical but swapping labels, we create
pure semantic drift where P(E) is stable but P(E|Y) changes.

- Global MMD: Should be ~0 (identical distributions)
- Conditional MMD: Should increase with swap percentage

Comparison: Verify if the "50% Swap" score is similar to the IMDb score (~0.011)
or if the different domain changes the scale.
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
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


def main():
    print("=" * 60)
    print("AMAZON CONCEPT SWAP EXPERIMENT")
    print("Cross-Domain Validation of Semantic Drift Detection")
    print("=" * 60)
    
    # Setup paths
    base_path = project_root
    embeddings_path = base_path / 'data' / 'processed' / 'amazon_embeddings.pt'
    output_path = base_path / 'results' / 'amazon_concept_swap.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load Amazon embeddings
    print("\nLoading Amazon embeddings...")
    data = load_embeddings(embeddings_path)
    all_embeddings = data['embeddings']
    all_labels = data['labels']
    
    print(f"Total samples: {all_embeddings.shape[0]}")
    
    # Select random subset of 3000 samples for Reference Set
    REF_SIZE = 3000
    torch.manual_seed(42)
    idx = torch.randperm(all_embeddings.shape[0])[:REF_SIZE]
    X_ref = all_embeddings[idx]
    Y_ref = all_labels[idx]
    
    print(f"Reference Set: {X_ref.shape[0]} samples")
    ref_pos, ref_neg = split_by_label(X_ref, Y_ref)
    print(f"  Positive: {ref_pos.shape[0]}")
    print(f"  Negative: {ref_neg.shape[0]}")
    
    # Create Stream Set - exact clone of Reference (perfect marginal stability)
    X_stream = X_ref.clone()
    print(f"\nStream Set: {X_stream.shape[0]} samples (exact clone)")
    print("  → P(E) is PERFECTLY STABLE")
    
    # Swap percentages to test
    swap_percentages = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    
    global_mmd_scores = []
    cond_mmd_scores = []
    
    # For MMD computation, use small subsamples for speed
    SAMPLE_SIZE = 200
    
    print("\nRunning swap experiment...")
    print("-" * 50)
    
    with torch.no_grad():
        for p in swap_percentages:
            print(f"\nSwap Percentage: {p*100:.0f}%")
            
            # Create corrupted labels
            Y_corrupt = flip_labels(Y_ref, p, seed=42)
            n_flipped = (Y_corrupt != Y_ref).sum().item()
            print(f"  Labels flipped: {n_flipped}")
            
            # Metric 1: Global MMD (should be ~0)
            # Subsample for efficiency
            sample_idx = torch.randperm(REF_SIZE)[:SAMPLE_SIZE]
            ref_sample = X_ref[sample_idx]
            stream_sample = X_stream[sample_idx]
            
            global_mmd = mmd_rbf(ref_sample, stream_sample, chunk_size=128).item()
            global_mmd_scores.append(global_mmd)
            print(f"  Global MMD: {global_mmd:.6f}")
            
            # Metric 2: Conditional MMD
            # Split reference by ground truth labels
            ref_pos, ref_neg = split_by_label(X_ref, Y_ref)
            
            # Split stream by corrupted labels
            stream_pos, stream_neg = split_by_label(X_stream, Y_corrupt)
            
            # Subsample for efficiency
            if ref_pos.shape[0] > SAMPLE_SIZE:
                ref_pos_sample = ref_pos[torch.randperm(ref_pos.shape[0])[:SAMPLE_SIZE]]
            else:
                ref_pos_sample = ref_pos
                
            if ref_neg.shape[0] > SAMPLE_SIZE:
                ref_neg_sample = ref_neg[torch.randperm(ref_neg.shape[0])[:SAMPLE_SIZE]]
            else:
                ref_neg_sample = ref_neg
                
            if stream_pos.shape[0] > SAMPLE_SIZE:
                stream_pos_sample = stream_pos[torch.randperm(stream_pos.shape[0])[:SAMPLE_SIZE]]
            else:
                stream_pos_sample = stream_pos
                
            if stream_neg.shape[0] > SAMPLE_SIZE:
                stream_neg_sample = stream_neg[torch.randperm(stream_neg.shape[0])[:SAMPLE_SIZE]]
            else:
                stream_neg_sample = stream_neg
            
            # Compute class-conditional MMD
            mmd_pos = mmd_rbf(ref_pos_sample, stream_pos_sample, chunk_size=128).item()
            mmd_neg = mmd_rbf(ref_neg_sample, stream_neg_sample, chunk_size=128).item()
            cond_mmd = 0.5 * mmd_pos + 0.5 * mmd_neg
            cond_mmd_scores.append(cond_mmd)
            
            print(f"  MMD (Positive class): {mmd_pos:.6f}")
            print(f"  MMD (Negative class): {mmd_neg:.6f}")
            print(f"  Conditional MMD: {cond_mmd:.6f}")
    
    # Plotting
    print("\n" + "=" * 60)
    print("Generating plot...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Convert to percentages for x-axis
    x_values = [p * 100 for p in swap_percentages]
    
    # Plot Global MMD (Blue, Dashed) - expected to stay near zero
    ax.plot(
        x_values, 
        global_mmd_scores, 
        color='blue', 
        linestyle='--', 
        linewidth=2.5,
        marker='s',
        markersize=10,
        label='Global MMD (Marginal)'
    )
    
    # Plot Conditional MMD (Red, Solid) - expected to increase
    ax.plot(
        x_values, 
        cond_mmd_scores, 
        color='red', 
        linestyle='-', 
        linewidth=2.5,
        marker='o',
        markersize=10,
        label='Conditional MMD (Semantic)'
    )
    
    # Title and labels
    ax.set_title('Cross-Domain Validation: Amazon Electronics (DistilBERT)', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Label Swap Percentage (%)', fontsize=12)
    ax.set_ylabel('MMD Score', fontsize=12)
    
    # Grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=11)
    
    # Set x-axis ticks
    ax.set_xticks(x_values)
    
    # Tight layout
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Plot saved to: {output_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"\n{'Swap %':<10} {'Global MMD':<15} {'Conditional MMD':<15}")
    print("-" * 40)
    for i, p in enumerate(swap_percentages):
        print(f"{p*100:<10.0f} {global_mmd_scores[i]:<15.6f} {cond_mmd_scores[i]:<15.6f}")
    
    print(f"\n📊 Key Observations:")
    print(f"   Global MMD range: {min(global_mmd_scores):.6f} to {max(global_mmd_scores):.6f}")
    print(f"   Cond. MMD range:  {min(cond_mmd_scores):.6f} to {max(cond_mmd_scores):.6f}")
    print(f"   Cond. MMD increase: {cond_mmd_scores[-1] - cond_mmd_scores[0]:.6f}")
    
    # Cross-domain comparison
    imdb_50_swap_cond_mmd = 0.011  # Approximate value from IMDb experiment
    amazon_50_swap_cond_mmd = cond_mmd_scores[-1]
    
    print(f"\n🔄 CROSS-DOMAIN COMPARISON (50% Swap):")
    print(f"   IMDb Conditional MMD:   ~{imdb_50_swap_cond_mmd:.6f}")
    print(f"   Amazon Conditional MMD:  {amazon_50_swap_cond_mmd:.6f}")
    
    ratio = amazon_50_swap_cond_mmd / imdb_50_swap_cond_mmd
    if 0.5 < ratio < 2.0:
        print(f"   → Similar scale (ratio: {ratio:.2f}x)")
        print(f"   → Detector shows consistent sensitivity across domains!")
    else:
        print(f"   → Different scale (ratio: {ratio:.2f}x)")
        print(f"   → Domain-specific calibration may be needed.")
    
    if cond_mmd_scores[-1] > global_mmd_scores[-1] * 2:
        print(f"\n✅ HYPOTHESIS VALIDATED: Conditional MMD detects semantic drift")
        print(f"   while Global MMD remains stable on Amazon data!")


if __name__ == '__main__':
    main()
