"""
Visualize drift detection results.

This script creates a plot comparing Global (Marginal) MMD vs
Conditional (Semantic) MMD over time to visualize drift patterns.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    # Define paths
    base_path = Path(__file__).parent.parent
    input_path = base_path / 'results' / 'imdb_drift_results.csv'
    output_path = base_path / 'results' / 'drift_plot.png'
    
    # Load data
    print(f"Loading results from: {input_path}")
    df = pd.read_csv(input_path)
    
    # Preprocessing
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    print(f"Loaded {len(df)} data points")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    # Smoothing: Rolling mean with window=50
    df['Global_Smooth'] = df['Global_MMD'].rolling(window=50, center=True).mean()
    df['Cond_Smooth'] = df['Conditional_MMD'].rolling(window=50, center=True).mean()
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot Global MMD (Marginal) - Blue, Dashed
    ax.plot(
        df['Date'], 
        df['Global_Smooth'], 
        color='blue', 
        linestyle='--', 
        linewidth=2,
        label='Global MMD (Marginal)'
    )
    
    # Plot Conditional MMD (Semantic) - Red, Solid
    ax.plot(
        df['Date'], 
        df['Cond_Smooth'], 
        color='red', 
        linestyle='-', 
        linewidth=2,
        label='Conditional MMD (Semantic)'
    )
    
    # Title and labels
    ax.set_title('Longitudinal Drift Detection: Marginal vs. Semantic (IMDb)', fontsize=14, fontweight='bold')
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
    print("Plot saved!")
    
    # Also display some statistics
    print("\n" + "=" * 50)
    print("SUMMARY STATISTICS")
    print("=" * 50)
    print(f"Global MMD  - Mean: {df['Global_MMD'].mean():.6f}, Std: {df['Global_MMD'].std():.6f}")
    print(f"Cond. MMD   - Mean: {df['Conditional_MMD'].mean():.6f}, Std: {df['Conditional_MMD'].std():.6f}")


if __name__ == '__main__':
    main()
