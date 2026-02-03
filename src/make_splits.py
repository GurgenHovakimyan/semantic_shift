"""
Split processed IMDb data into reference and stream sets.

This script divides the cleaned data into two experimental sets:
- Reference Set (2005-2010): "Gold Standard" training era
- Stream Set (2013-2018): Deployment era for drift testing

A gap of 2011-2012 is intentionally created between the sets.
"""

import pandas as pd
from pathlib import Path


def load_processed_data(filepath: Path) -> pd.DataFrame:
    """Load the processed IMDb data."""
    print(f"Loading processed data from: {filepath}")
    df = pd.read_pickle(filepath)
    print(f"Loaded {len(df)} records")
    return df


def split_by_year(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into reference and stream sets based on year.
    
    Reference Set: 2005-2010 (inclusive)
    Stream Set: 2013-2018 (inclusive)
    Gap: 2011-2012 (excluded from both sets)
    """
    # Extract year from review_date
    df['year'] = df['review_date'].dt.year
    
    # Reference set: 2005-2010
    df_ref = df[(df['year'] >= 2005) & (df['year'] <= 2010)].copy()
    
    # Stream set: 2013-2018
    df_stream = df[(df['year'] >= 2013) & (df['year'] <= 2018)].copy()
    
    return df_ref, df_stream


def balance_reference_set(df_ref: pd.DataFrame, max_samples: int = 50000) -> pd.DataFrame:
    """
    Downsample the reference set if it exceeds max_samples.
    
    Args:
        df_ref: Reference DataFrame
        max_samples: Maximum number of samples to keep
        
    Returns:
        Balanced reference DataFrame
    """
    if len(df_ref) > max_samples:
        print(f"Reference set is large ({len(df_ref)} samples). Downsampling to {max_samples}...")
        df_ref = df_ref.sample(n=max_samples, random_state=42).reset_index(drop=True)
    return df_ref


def main():
    # Define paths
    base_path = Path(__file__).parent.parent
    input_path = base_path / 'data' / 'processed' / 'imdb_clean.pkl'
    ref_output_path = base_path / 'data' / 'processed' / 'reference_set.pkl'
    stream_output_path = base_path / 'data' / 'processed' / 'stream_set.pkl'
    
    # Load processed data
    df = load_processed_data(input_path)
    
    # Split by year
    print("\nSplitting data by year...")
    df_ref, df_stream = split_by_year(df)
    
    print(f"Reference set (2005-2010): {len(df_ref)} samples")
    print(f"Stream set (2013-2018): {len(df_stream)} samples")
    
    # Balance reference set if needed
    df_ref = balance_reference_set(df_ref)
    
    # Save the splits
    df_ref.to_pickle(ref_output_path)
    print(f"\nSaved reference set to: {ref_output_path}")
    
    df_stream.to_pickle(stream_output_path)
    print(f"Saved stream set to: {stream_output_path}")
    
    # Verification
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    print("\n📊 REFERENCE SET:")
    print(f"   Shape: {df_ref.shape}")
    print(f"   Year range: {df_ref['year'].min()} to {df_ref['year'].max()}")
    print(f"   Label distribution:")
    print(f"      Positive (1): {(df_ref['label'] == 1).sum()}")
    print(f"      Negative (0): {(df_ref['label'] == 0).sum()}")
    
    print("\n📊 STREAM SET:")
    print(f"   Shape: {df_stream.shape}")
    print(f"   Year range: {df_stream['year'].min()} to {df_stream['year'].max()}")
    print(f"   Label distribution:")
    print(f"      Positive (1): {(df_stream['label'] == 1).sum()}")
    print(f"      Negative (0): {(df_stream['label'] == 0).sum()}")
    
    print("\n✅ Gap confirmed: Years 2011-2012 are excluded from both sets.")


if __name__ == '__main__':
    main()
