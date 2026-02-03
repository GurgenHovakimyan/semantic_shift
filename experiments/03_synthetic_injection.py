"""
Synthetic Injection Experiment.

This experiment proves that Conditional MMD can detect semantic drift
even when Global MMD fails to detect it.

The key insight: By rotating positive class embeddings toward the negative
class center, we change P(Y|E) while keeping P(E) roughly stable.
- Global MMD (measures P(E)) stays flat
- Conditional MMD (measures P(E|Y)) shoots up
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.mmd import mmd_rbf_chunked as mmd_rbf


def load_reference_embeddings(filepath: Path) -> dict:
    """Load reference embeddings."""
    data = torch.load(filepath, map_location='cpu', weights_only=False)
    return data


def split_by_label(embeddings: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split embeddings into positive (1) and negative (0) subsets."""
    pos_mask = labels == 1
    neg_mask = labels == 0
    return embeddings[pos_mask], embeddings[neg_mask]


def rotate_toward_center(embeddings: torch.Tensor, source_center: torch.Tensor, 
                         target_center: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Rotate embeddings toward a target center while preserving variance.
    
    Instead of collapsing to target, we shift the center while keeping
    the relative structure of the embeddings.
    
    Args:
        embeddings: Original embeddings (N, D)
        source_center: Current center of embeddings (D,)
        target_center: Target center to rotate toward (D,)
        alpha: Interpolation factor (0 = no change, 1 = centered at target)
        
    Returns:
        Rotated embeddings with preserved variance
    """
    # Compute the shift direction
    shift_direction = target_center - source_center
    
    # Shift all embeddings by alpha * shift_direction
    # This moves the center but preserves the cloud structure
    rotated = embeddings + alpha * shift_direction.unsqueeze(0)
    return rotated


def main():
    print("=" * 60)
    print("SYNTHETIC INJECTION EXPERIMENT")
    print("Proving Conditional MMD > Global MMD for Semantic Drift")
    print("=" * 60)
    
    # Setup paths
    base_path = project_root
    ref_path = base_path / 'data' / 'processed' / 'ref_embeddings.pt'
    output_path = base_path / 'results' / 'synthetic_drift_plot.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load reference embeddings
    print("\nLoading reference embeddings...")
    ref_data = load_reference_embeddings(ref_path)
    ref_embeddings = ref_data['embeddings']
    ref_labels = ref_data['labels']
    
    print(f"Reference: {ref_embeddings.shape[0]} samples")
    
    # Split reference by label
    ref_pos, ref_neg = split_by_label(ref_embeddings, ref_labels)
    print(f"Reference Positive: {ref_pos.shape[0]} samples")
    print(f"Reference Negative: {ref_neg.shape[0]} samples")
    
    # Compute class centers
    neg_center = ref_neg.mean(dim=0)
    pos_center = ref_pos.mean(dim=0)
    print(f"Computed class centers (dim={neg_center.shape[0]})")
    
    # Create "Stable Stream" - random sample of 2000 from reference
    STREAM_SIZE = 2000
    torch.manual_seed(42)
    idx = torch.randperm(ref_embeddings.shape[0])[:STREAM_SIZE]
    stream_embeddings = ref_embeddings[idx].clone()
    stream_labels = ref_labels[idx].clone()
    
    stream_pos, stream_neg = split_by_label(stream_embeddings, stream_labels)
    print(f"\nStable Stream: {STREAM_SIZE} samples")
    print(f"  Positive: {stream_pos.shape[0]}")
    print(f"  Negative: {stream_neg.shape[0]}")
    
    # Sample reference for MMD computation (small sizes for fast computation)
    REF_SAMPLE_SIZE = 200  # Reduced for speed
    ref_sample_idx = torch.randperm(ref_embeddings.shape[0])[:REF_SAMPLE_SIZE]
    ref_sample = ref_embeddings[ref_sample_idx]
    
    ref_pos_sample_idx = torch.randperm(ref_pos.shape[0])[:REF_SAMPLE_SIZE]
    ref_pos_sample = ref_pos[ref_pos_sample_idx]
    
    ref_neg_sample_idx = torch.randperm(ref_neg.shape[0])[:REF_SAMPLE_SIZE]
    ref_neg_sample = ref_neg[ref_neg_sample_idx]
    
    # Injection Loop: 10 steps of increasing drift
    print("\nRunning injection experiment...")
    alphas = np.linspace(0.0, 1.0, 11)  # 0.0, 0.1, 0.2, ..., 1.0
    
    global_mmd_scores = []
    cond_mmd_scores = []

    
    # Get original stream positive embeddings (before any rotation)
    stream_pos_original = stream_pos.clone()
    stream_pos_center = stream_pos_original.mean(dim=0)
    
    # Also subsample stream_neg to avoid memory issues
    if stream_neg.shape[0] > REF_SAMPLE_SIZE:
        neg_sample_idx = torch.randperm(stream_neg.shape[0])[:REF_SAMPLE_SIZE]
        stream_neg_sample = stream_neg[neg_sample_idx]
    else:
        stream_neg_sample = stream_neg
    
    with torch.no_grad():
        for alpha in tqdm(alphas, desc="Injection Steps"):
            # Rotate positive class toward negative center (preserving variance)
            # This simulates sentiment flip: positive reviews becoming more like negative
            stream_pos_rotated = rotate_toward_center(
                stream_pos_original, stream_pos_center, neg_center, alpha
            )
            
            # Reconstruct stream with rotated positives and unchanged negatives
            stream_modified = torch.cat([stream_pos_rotated, stream_neg], dim=0)
            
            # Sample for MMD computation
            if stream_modified.shape[0] > REF_SAMPLE_SIZE:
                sample_idx = torch.randperm(stream_modified.shape[0])[:REF_SAMPLE_SIZE]
                stream_sample = stream_modified[sample_idx]
            else:
                stream_sample = stream_modified
            
            # Metric 1: Global MMD (should stay flat - P(E) is stable)
            global_mmd = mmd_rbf(ref_sample, stream_sample).item()
            global_mmd_scores.append(global_mmd)
            
            # Metric 2: Conditional MMD (should increase - P(E|Y) is changing)
            # Compare rotated positives with reference positives
            if stream_pos_rotated.shape[0] > REF_SAMPLE_SIZE:
                pos_sample_idx = torch.randperm(stream_pos_rotated.shape[0])[:REF_SAMPLE_SIZE]
                stream_pos_sample = stream_pos_rotated[pos_sample_idx]
            else:
                stream_pos_sample = stream_pos_rotated
                
            mmd_pos = mmd_rbf(ref_pos_sample, stream_pos_sample).item()
            mmd_neg = mmd_rbf(ref_neg_sample, stream_neg_sample).item()  # Use subsampled negatives
            cond_mmd = 0.5 * mmd_pos + 0.5 * mmd_neg
            cond_mmd_scores.append(cond_mmd)
            
            print(f"  α={alpha:.1f}: Global MMD={global_mmd:.4f}, Conditional MMD={cond_mmd:.4f}")
    
    # Plotting
    print("\nGenerating plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot Global MMD (Blue, Dashed) - expected to stay flat
    ax.plot(
        alphas, 
        global_mmd_scores, 
        color='blue', 
        linestyle='--', 
        linewidth=2.5,
        marker='o',
        markersize=8,
        label='Global MMD (Marginal)'
    )
    
    # Plot Conditional MMD (Red, Solid) - expected to shoot up
    ax.plot(
        alphas, 
        cond_mmd_scores, 
        color='red', 
        linestyle='-', 
        linewidth=2.5,
        marker='s',
        markersize=8,
        label='Conditional MMD (Semantic)'
    )
    
    # Title and labels
    ax.set_title('Synthetic Drift Injection: Global vs. Conditional MMD', fontsize=14, fontweight='bold')
    ax.set_xlabel('Drift Intensity (α)', fontsize=12)
    ax.set_ylabel('MMD Score', fontsize=12)
    
    # Grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=11)
    
    # Set x-axis ticks
    ax.set_xticks(alphas)
    ax.set_xticklabels([f'{a:.1f}' for a in alphas])
    
    # Tight layout
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Plot saved to: {output_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"Alpha Range: {alphas[0]:.1f} to {alphas[-1]:.1f}")
    print(f"\nGlobal MMD:")
    print(f"  Start (α=0): {global_mmd_scores[0]:.4f}")
    print(f"  End (α=1):   {global_mmd_scores[-1]:.4f}")
    print(f"  Change:      {global_mmd_scores[-1] - global_mmd_scores[0]:.4f}")
    print(f"\nConditional MMD:")
    print(f"  Start (α=0): {cond_mmd_scores[0]:.4f}")
    print(f"  End (α=1):   {cond_mmd_scores[-1]:.4f}")
    print(f"  Change:      {cond_mmd_scores[-1] - cond_mmd_scores[0]:.4f}")
    print(f"\n📊 Ratio of change (Cond/Global): {(cond_mmd_scores[-1] - cond_mmd_scores[0]) / max(0.0001, global_mmd_scores[-1] - global_mmd_scores[0]):.2f}x")


if __name__ == '__main__':
    main()
