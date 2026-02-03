"""
Preprocess raw IMDb reviews data.

This script cleans and processes the raw IMDB_reviews.json file,
extracting relevant fields, filtering by date and rating, and
creating binary sentiment labels.
"""

import json
import pandas as pd
from pathlib import Path


def load_raw_reviews(filepath: Path) -> list[dict]:
    """Load line-by-line JSON file and extract relevant keys."""
    records = []
    keys_to_extract = ['review_date', 'review_summary', 'review_text', 'rating', 'movie_id']
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                record = {key: data.get(key) for key in keys_to_extract}
                records.append(record)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping malformed JSON line: {e}")
                continue
    
    return records


def preprocess_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the reviews DataFrame.
    
    Steps:
    1. Create full_text column by concatenating review_summary and review_text
    2. Convert review_date to datetime
    3. Filter by date range (2005-2023)
    4. Drop rows with missing ratings
    5. Create binary label column (exclude neutral ratings 5-6)
    """
    # Create full_text column
    df['full_text'] = df['review_summary'].fillna('') + ' ' + df['review_text'].fillna('')
    df['full_text'] = df['full_text'].str.strip()
    
    # Convert review_date to datetime
    df['review_date'] = pd.to_datetime(df['review_date'], errors='coerce')
    
    # Convert rating to numeric
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    
    # Filter by date range (2005-2023 inclusive)
    df = df[
        (df['review_date'].dt.year >= 2005) & 
        (df['review_date'].dt.year <= 2023)
    ].copy()
    
    # Drop rows where rating is missing
    df = df.dropna(subset=['rating']).copy()
    
    # Create binary label column
    # 1 (Positive) if rating > 6
    # 0 (Negative) if rating < 5
    # Exclude neutral ratings (5 and 6)
    df['label'] = None
    df.loc[df['rating'] > 6, 'label'] = 1
    df.loc[df['rating'] < 5, 'label'] = 0
    
    # Filter out neutral ratings (keep only rows with label assigned)
    df = df[df['label'].notna()].copy()
    df['label'] = df['label'].astype(int)
    
    # Reset index
    df = df.reset_index(drop=True)
    
    return df


def main():
    # Define paths
    base_path = Path(__file__).parent.parent
    input_path = base_path / 'data' / 'raw' / 'IMDB_reviews' / 'IMDB_reviews.json'
    output_path = base_path / 'data' / 'processed' / 'imdb_clean.pkl'
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading raw reviews from: {input_path}")
    
    # Load raw data
    records = load_raw_reviews(input_path)
    print(f"Loaded {len(records)} raw records")
    
    # Create DataFrame
    df = pd.DataFrame(records)
    
    # Preprocess
    print("Preprocessing reviews...")
    df_clean = preprocess_reviews(df)
    
    # Save to pickle
    df_clean.to_pickle(output_path)
    print(f"Saved cleaned data to: {output_path}")
    
    # Verification: Print statistics
    print("\n" + "=" * 50)
    print("VERIFICATION SUMMARY")
    print("=" * 50)
    print(f"Total count of reviews: {len(df_clean)}")
    print(f"Date range: {df_clean['review_date'].min()} to {df_clean['review_date'].max()}")
    print(f"\nLabel distribution:")
    print(df_clean['label'].value_counts().to_string())
    print(f"\nPositive reviews (label=1): {(df_clean['label'] == 1).sum()}")
    print(f"Negative reviews (label=0): {(df_clean['label'] == 0).sum()}")


if __name__ == '__main__':
    main()
