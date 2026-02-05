"""
Qualitative Analysis: Top Keywords Driving Semantic Drift

This experiment identifies the top keywords that characterize semantic drift
by comparing word frequencies between old and new time periods.

Methodology:
- Use CountVectorizer to extract word frequencies
- Compare average frequencies between old and new periods
- Compute Drift Score = F_new - F_old
- Identify "New Concepts" (positive drift) and "Old Concepts" (negative drift)

Focus: Amazon Electronics dataset (has reliable timestamps)
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


def extract_top_drift_terms(texts_old: list, texts_new: list, 
                            top_n: int = 20, min_df: int = 10) -> dict:
    """
    Extract top keywords driving semantic drift between two text corpora.
    
    Methodology:
    1. Fit CountVectorizer on combined corpus
    2. Calculate average word frequency for each corpus
    3. Compute Drift Score = F_new - F_old
    4. Return top positive (new concepts) and negative (old concepts) scores
    
    Args:
        texts_old: List of texts from the older time period
        texts_new: List of texts from the newer time period
        top_n: Number of top terms to return for each direction
        min_df: Minimum document frequency for terms
        
    Returns:
        Dictionary with 'new_concepts' and 'old_concepts' DataFrames
    """
    print(f"\nAnalyzing {len(texts_old)} old texts vs {len(texts_new)} new texts...")
    
    # Initialize CountVectorizer
    vectorizer = CountVectorizer(
        stop_words='english',
        min_df=min_df,
        max_features=10000,  # Limit vocabulary size
        lowercase=True,
        token_pattern=r'(?u)\b[a-zA-Z]{3,}\b'  # Words with 3+ letters only
    )
    
    # Fit on combined corpus
    all_texts = texts_old + texts_new
    print(f"Fitting vectorizer on {len(all_texts)} total documents...")
    vectorizer.fit(all_texts)
    
    vocab = vectorizer.get_feature_names_out()
    print(f"Vocabulary size: {len(vocab)} terms")
    
    # Transform each corpus
    X_old = vectorizer.transform(texts_old)
    X_new = vectorizer.transform(texts_new)
    
    # Calculate average word frequency per document
    # Normalize by number of documents to get comparable frequencies
    F_old = np.array(X_old.mean(axis=0)).flatten()
    F_new = np.array(X_new.mean(axis=0)).flatten()
    
    # Compute Drift Score: F_new - F_old
    drift_scores = F_new - F_old
    
    # Also compute log odds ratio for additional insight
    # Add small epsilon to avoid division by zero
    epsilon = 1e-8
    log_odds = np.log((F_new + epsilon) / (F_old + epsilon))
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'term': vocab,
        'freq_old': F_old,
        'freq_new': F_new,
        'drift_score': drift_scores,
        'log_odds': log_odds
    })
    
    # Sort by drift score
    results_df = results_df.sort_values('drift_score', ascending=False)
    
    # Extract top new concepts (highest positive drift)
    new_concepts = results_df.head(top_n).copy()
    new_concepts['direction'] = 'NEW'
    
    # Extract top old concepts (highest negative drift = lowest values)
    old_concepts = results_df.tail(top_n).sort_values('drift_score').copy()
    old_concepts['direction'] = 'OLD'
    
    return {
        'new_concepts': new_concepts,
        'old_concepts': old_concepts,
        'full_results': results_df
    }


def print_drift_table(results: dict, dataset_name: str):
    """Pretty print the drift analysis results."""
    print("\n" + "=" * 70)
    print(f"TOP DRIFT TERMS: {dataset_name}")
    print("=" * 70)
    
    # New Concepts Table
    print("\n📈 NEW CONCEPTS (Rising in newer period)")
    print("-" * 60)
    print(f"{'Rank':<6}{'Term':<20}{'Freq (Old)':<12}{'Freq (New)':<12}{'Drift Score':<12}")
    print("-" * 60)
    
    for i, (_, row) in enumerate(results['new_concepts'].iterrows(), 1):
        print(f"{i:<6}{row['term']:<20}{row['freq_old']:<12.4f}{row['freq_new']:<12.4f}{row['drift_score']:<12.4f}")
    
    # Old Concepts Table
    print("\n📉 OLD CONCEPTS (Declining in newer period)")
    print("-" * 60)
    print(f"{'Rank':<6}{'Term':<20}{'Freq (Old)':<12}{'Freq (New)':<12}{'Drift Score':<12}")
    print("-" * 60)
    
    for i, (_, row) in enumerate(results['old_concepts'].iterrows(), 1):
        print(f"{i:<6}{row['term']:<20}{row['freq_old']:<12.4f}{row['freq_new']:<12.4f}{row['drift_score']:<12.4f}")


def analyze_amazon():
    """Run qualitative analysis on Amazon Electronics dataset."""
    print("=" * 70)
    print("QUALITATIVE ANALYSIS: Amazon Electronics Reviews")
    print("=" * 70)
    
    # Load Amazon data
    data_path = project_root / 'data' / 'processed' / 'amazon_electronics_balanced.pkl'
    
    if not data_path.exists():
        print(f"ERROR: Amazon data not found at {data_path}")
        print("Please run: python src/data_loader_amazon.py")
        return None
    
    print(f"\nLoading Amazon data from {data_path}...")
    df = pd.read_pickle(data_path)
    
    print(f"Total samples: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    # Check for date column
    date_col = None
    for col in ['review_date', 'date', 'timestamp', 'review_time']:
        if col in df.columns:
            date_col = col
            break
    
    if date_col is None:
        print("WARNING: No date column found. Using index order (assuming chronological).")
        df = df.reset_index(drop=True)
        df['_order'] = df.index
        date_col = '_order'
    else:
        # Sort by date
        print(f"Sorting by {date_col}...")
        df = df.sort_values(date_col).reset_index(drop=True)
    
    # Display date range if available
    if date_col != '_order':
        print(f"Date range: {df[date_col].min()} to {df[date_col].max()}")
    
    # Split: Oldest 3000 vs Newest 3000
    N_SAMPLES = 3000
    
    df_old = df.head(N_SAMPLES)
    df_new = df.tail(N_SAMPLES)
    
    print(f"\nOLD period: First {N_SAMPLES} reviews")
    if date_col != '_order':
        print(f"  Date range: {df_old[date_col].min()} to {df_old[date_col].max()}")
    
    print(f"\nNEW period: Last {N_SAMPLES} reviews")
    if date_col != '_order':
        print(f"  Date range: {df_new[date_col].min()} to {df_new[date_col].max()}")
    
    # Get text column
    text_col = None
    for col in ['text', 'review_text', 'review_body', 'content']:
        if col in df.columns:
            text_col = col
            break
    
    if text_col is None:
        print("ERROR: No text column found!")
        print(f"Available columns: {list(df.columns)}")
        return None
    
    print(f"\nUsing text column: '{text_col}'")
    
    # Extract texts
    texts_old = df_old[text_col].dropna().tolist()
    texts_new = df_new[text_col].dropna().tolist()
    
    # Run extraction
    results = extract_top_drift_terms(texts_old, texts_new)
    
    # Print results
    print_drift_table(results, "AMAZON ELECTRONICS")
    
    return results


def analyze_imdb():
    """Run qualitative analysis on IMDb Reviews dataset."""
    print("\n" + "=" * 70)
    print("QUALITATIVE ANALYSIS: IMDb Movie Reviews")
    print("=" * 70)
    
    # Load IMDb data
    data_path = project_root / 'data' / 'processed' / 'imdb_clean.pkl'
    
    if not data_path.exists():
        print(f"ERROR: IMDb data not found at {data_path}")
        print("Please run: python src/preprocess.py")
        return None
    
    print(f"\nLoading IMDb data from {data_path}...")
    df = pd.read_pickle(data_path)
    
    print(f"Total samples: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    # Sort by date
    date_col = 'review_date'
    print(f"Sorting by {date_col}...")
    df = df.sort_values(date_col).reset_index(drop=True)
    
    print(f"Date range: {df[date_col].min()} to {df[date_col].max()}")
    
    # Split: Oldest 3000 vs Newest 3000
    N_SAMPLES = 3000
    
    df_old = df.head(N_SAMPLES)
    df_new = df.tail(N_SAMPLES)
    
    print(f"\nOLD period: First {N_SAMPLES} reviews")
    print(f"  Date range: {df_old[date_col].min()} to {df_old[date_col].max()}")
    
    print(f"\nNEW period: Last {N_SAMPLES} reviews")
    print(f"  Date range: {df_new[date_col].min()} to {df_new[date_col].max()}")
    
    # Use review_text column
    text_col = 'review_text'
    print(f"\nUsing text column: '{text_col}'")
    
    # Extract texts
    texts_old = df_old[text_col].dropna().tolist()
    texts_new = df_new[text_col].dropna().tolist()
    
    # Run extraction
    results = extract_top_drift_terms(texts_old, texts_new)
    
    # Print results
    print_drift_table(results, "IMDb MOVIE REVIEWS")
    
    return results


def main():
    print("\n" + "=" * 70)
    print("QUALITATIVE DRIFT ANALYSIS")
    print("Identifying Keywords Driving Semantic Shift")
    print("=" * 70)
    
    # Analyze Amazon (has reliable timestamps)
    amazon_results = analyze_amazon()
    
    if amazon_results is not None:
        # Save results to CSV
        output_path = project_root / 'results' / 'amazon_drift_terms.csv'
        
        # Combine new and old concepts
        combined = pd.concat([
            amazon_results['new_concepts'],
            amazon_results['old_concepts']
        ])
        combined.to_csv(output_path, index=False)
        print(f"\n✅ Amazon results saved to: {output_path}")
    
    # Analyze IMDb (has dates in preprocessed data)
    imdb_results = analyze_imdb()
    
    if imdb_results is not None:
        # Save results to CSV
        output_path = project_root / 'results' / 'imdb_drift_terms.csv'
        
        # Combine new and old concepts
        combined = pd.concat([
            imdb_results['new_concepts'],
            imdb_results['old_concepts']
        ])
        combined.to_csv(output_path, index=False)
        print(f"\n✅ IMDb results saved to: {output_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("""
📊 How to Read the Results:

1. NEW CONCEPTS (Rising Terms):
   - Words appearing MORE frequently in recent reviews
   - May indicate: new products, technologies, features, or trends
   - Example: 'wireless', 'bluetooth', 'usb-c' (if rising)

2. OLD CONCEPTS (Declining Terms):
   - Words appearing LESS frequently in recent reviews  
   - May indicate: outdated products, technologies, or features
   - Example: 'dvd', 'vhs', 'cassette' (if declining)

3. Drift Score Interpretation:
   - Positive = term is MORE common in new period
   - Negative = term is MORE common in old period
   - Magnitude = strength of the drift

4. Scientific Significance:
   - These terms represent the LEXICAL dimension of semantic drift
   - Complements our EMBEDDING-based (Conditional MMD) detection
   - Provides interpretable evidence of what concepts are changing
""")
    
    print("\n" + "=" * 70)
    print("Qualitative Analysis Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
