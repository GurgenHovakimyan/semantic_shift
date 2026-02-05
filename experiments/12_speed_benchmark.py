"""
Speed Benchmark: MMD vs C2ST Classifier Comparison.

This experiment measures and compares execution time of:
- MMD (Our Method): Maximum Mean Discrepancy with RBF kernel
- C2ST Classifier (Competitor): Logistic Regression classifier

Goal: Demonstrate computational efficiency of MMD over classifier-based approaches.

Datasets:
- IMDb: data/processed/ref_embeddings.pt (DistilBERT)
- Amazon: data/processed/amazon_embeddings.pt (DistilBERT)
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from src.mmd import mmd_rbf_chunked as mmd_rbf


def load_embeddings(filepath: Path) -> dict:
    """Load embeddings from file."""
    data = torch.load(filepath, map_location='cpu', weights_only=False)
    return data


def benchmark_mmd(X_ref: torch.Tensor, X_stream: torch.Tensor, n_repeats: int = 50) -> dict:
    """
    Benchmark MMD computation time.
    
    Args:
        X_ref: Reference embeddings (N, D)
        X_stream: Stream embeddings (N, D)
        n_repeats: Number of repetitions for averaging
        
    Returns:
        Dictionary with timing statistics
    """
    times = []
    
    with torch.no_grad():
        for _ in range(n_repeats):
            start = time.perf_counter()
            _ = mmd_rbf(X_ref, X_stream, chunk_size=128)
            end = time.perf_counter()
            times.append(end - start)
    
    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times)
    }


def benchmark_c2st(X_ref: torch.Tensor, X_stream: torch.Tensor, n_repeats: int = 10) -> dict:
    """
    Benchmark C2ST Classifier (Logistic Regression) training time.
    
    C2ST (Classifier Two-Sample Test) trains a classifier to distinguish
    between samples from two distributions. If accuracy > 0.5, distributions differ.
    
    Args:
        X_ref: Reference embeddings (N, D)
        X_stream: Stream embeddings (N, D)
        n_repeats: Number of repetitions for averaging
        
    Returns:
        Dictionary with timing statistics
    """
    # Convert to numpy
    X_ref_np = X_ref.numpy()
    X_stream_np = X_stream.numpy()
    
    # Create combined dataset with labels
    X = np.vstack([X_ref_np, X_stream_np])
    y = np.concatenate([np.zeros(len(X_ref_np)), np.ones(len(X_stream_np))])
    
    times = []
    
    for _ in range(n_repeats):
        start = time.perf_counter()
        
        # Train Logistic Regression classifier
        clf = LogisticRegression(
            max_iter=1000,
            solver='lbfgs',
            n_jobs=1  # Single CPU for fair comparison
        )
        clf.fit(X, y)
        
        end = time.perf_counter()
        times.append(end - start)
    
    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times)
    }


def benchmark_dataset(name: str, embeddings: torch.Tensor, labels: torch.Tensor = None) -> pd.DataFrame:
    """
    Run full benchmark on a dataset.
    
    Args:
        name: Dataset name for reporting
        embeddings: Full embedding tensor
        labels: Optional labels (not used for timing, just for info)
        
    Returns:
        DataFrame with benchmark results
    """
    print(f"\n{'=' * 70}")
    print(f"BENCHMARKING: {name}")
    print(f"{'=' * 70}")
    print(f"Total samples available: {embeddings.shape[0]}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    
    # Sample sizes to test (keep small to avoid long runtimes)
    N_values = [100, 200, 500, 1000]
    
    # Filter N values that are feasible
    max_n = embeddings.shape[0] // 2
    N_values = [n for n in N_values if n <= max_n]
    
    print(f"Sample sizes to test: {N_values}")
    
    results = []
    
    for N in N_values:
        print(f"\n--- Sample Size N = {N} (2N = {2*N} total) ---")
        
        # Random sample
        torch.manual_seed(42)
        indices = torch.randperm(embeddings.shape[0])[:2*N]
        X_ref = embeddings[indices[:N]]
        X_stream = embeddings[indices[N:]]
        
        # Adjust repeats based on N (fewer for larger N)
        mmd_repeats = 5 if N >= 500 else 20
        c2st_repeats = 5
        
        # Benchmark MMD
        print(f"  Benchmarking MMD ({mmd_repeats} repeats)...", end=" ", flush=True)
        mmd_stats = benchmark_mmd(X_ref, X_stream, n_repeats=mmd_repeats)
        print(f"Done. Mean: {mmd_stats['mean']*1000:.2f}ms")
        
        # Benchmark C2ST
        print(f"  Benchmarking C2ST ({c2st_repeats} repeats)...", end=" ", flush=True)
        c2st_stats = benchmark_c2st(X_ref, X_stream, n_repeats=c2st_repeats)
        print(f"Done. Mean: {c2st_stats['mean']*1000:.2f}ms")
        
        # Speedup factor
        speedup = c2st_stats['mean'] / mmd_stats['mean']
        print(f"  → MMD is {speedup:.1f}x faster than C2ST")
        
        results.append({
            'Dataset': name,
            'N': N,
            'MMD_mean_ms': mmd_stats['mean'] * 1000,
            'MMD_std_ms': mmd_stats['std'] * 1000,
            'C2ST_mean_ms': c2st_stats['mean'] * 1000,
            'C2ST_std_ms': c2st_stats['std'] * 1000,
            'Speedup': speedup
        })
    
    return pd.DataFrame(results)


def main():
    print("=" * 70)
    print("SPEED BENCHMARK: MMD vs C2ST Classifier")
    print("=" * 70)
    print("\nThis experiment compares execution time of:")
    print("  • MMD (Our Method): Kernel-based two-sample test")
    print("  • C2ST (Competitor): Logistic Regression classifier")
    
    # Setup paths
    base_path = project_root
    imdb_path = base_path / 'data' / 'processed' / 'ref_embeddings.pt'
    amazon_path = base_path / 'data' / 'processed' / 'amazon_embeddings.pt'
    output_path = base_path / 'results' / 'speed_benchmark.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    # Benchmark IMDb
    if imdb_path.exists():
        print(f"\nLoading IMDb embeddings from {imdb_path}...")
        imdb_data = load_embeddings(imdb_path)
        imdb_results = benchmark_dataset(
            "IMDb", 
            imdb_data['embeddings'], 
            imdb_data.get('labels')
        )
        all_results.append(imdb_results)
    else:
        print(f"\nWARNING: IMDb embeddings not found at {imdb_path}")
    
    # Benchmark Amazon
    if amazon_path.exists():
        print(f"\nLoading Amazon embeddings from {amazon_path}...")
        amazon_data = load_embeddings(amazon_path)
        amazon_results = benchmark_dataset(
            "Amazon", 
            amazon_data['embeddings'], 
            amazon_data.get('labels')
        )
        all_results.append(amazon_results)
    else:
        print(f"\nWARNING: Amazon embeddings not found at {amazon_path}")
    
    # Combine and display results
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        
        print("\n" + "=" * 70)
        print("SUMMARY RESULTS")
        print("=" * 70)
        
        # Print formatted table
        print("\n{:<10} {:>8} {:>12} {:>12} {:>12} {:>12} {:>10}".format(
            "Dataset", "N", "MMD (ms)", "MMD σ", "C2ST (ms)", "C2ST σ", "Speedup"
        ))
        print("-" * 78)
        
        for _, row in combined_results.iterrows():
            print("{:<10} {:>8} {:>12.2f} {:>12.2f} {:>12.2f} {:>12.2f} {:>10.1f}x".format(
                row['Dataset'],
                row['N'],
                row['MMD_mean_ms'],
                row['MMD_std_ms'],
                row['C2ST_mean_ms'],
                row['C2ST_std_ms'],
                row['Speedup']
            ))
        
        # Save results
        combined_results.to_csv(output_path, index=False)
        print(f"\n✅ Results saved to: {output_path}")
        
        # Compute overall statistics
        print("\n" + "-" * 70)
        print("AGGREGATE STATISTICS")
        print("-" * 70)
        
        avg_speedup = combined_results['Speedup'].mean()
        min_speedup = combined_results['Speedup'].min()
        max_speedup = combined_results['Speedup'].max()
        
        print(f"  Average Speedup: {avg_speedup:.1f}x")
        print(f"  Min Speedup: {min_speedup:.1f}x")
        print(f"  Max Speedup: {max_speedup:.1f}x")
        
        # Scaling analysis
        print("\n" + "-" * 70)
        print("SCALING ANALYSIS")
        print("-" * 70)
        print("  MMD: O(N²) complexity but optimized with chunked computation")
        print("  C2ST: O(N × D × iterations) for Logistic Regression")
        print("  For typical NLP embeddings (D=768), MMD is significantly faster")
        
    print("\n" + "=" * 70)
    print("Speed Benchmark Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
