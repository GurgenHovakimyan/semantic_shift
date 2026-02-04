"""
Amazon Electronics Reviews Data Loader

Loads and processes electronics reviews from Amazon reviews datasets (Hugging Face).
Filters for electronics/pc categories or tech keywords, keeps binary sentiment (1 or 5 stars).
"""

from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
import random
import os

# Configuration
# Using McAuley Lab's Amazon Reviews 2023 dataset (Electronics category)
DATASET_NAME = "McAuley-Lab/Amazon-Reviews-2023"
SUBSET = "raw_review_Electronics"
TARGET_CATEGORIES = {"electronics", "pc"}
TECH_KEYWORDS = ["battery", "screen", "wifi", "bluetooth", "cable"]
TARGET_STARS = {1.0, 5.0}  # Ratings in this dataset are floats
TARGET_PER_CLASS = 5000  # Target samples per class for balanced dataset
OUTPUT_PATH = "data/processed/amazon_electronics_balanced.pkl"


def is_tech_related(text: str) -> bool:
    """Check if review text contains tech-related keywords."""
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in TECH_KEYWORDS)


def load_amazon_electronics_reviews() -> pd.DataFrame:
    """
    Load and filter Amazon electronics reviews from Hugging Face datasets.
    
    Returns:
        pd.DataFrame: Processed reviews with columns ['text', 'label', 'review_id', 'review_date']
    """
    print(f"Loading {DATASET_NAME} dataset ({SUBSET}, streaming mode)...")
    dataset = load_dataset(
        DATASET_NAME,
        SUBSET,
        split="full",
        streaming=True,
        trust_remote_code=True
    )
    
    processed_reviews = []
    count_pos = 0
    count_neg = 0
    
    total_target = TARGET_PER_CLASS * 2
    print(f"Extracting balanced dataset: {TARGET_PER_CLASS} per class ({total_target} total)...")
    with tqdm(total=total_target, desc="Processing reviews") as pbar:
        for sample in dataset:
            # Check if we've collected enough samples for both classes
            if count_pos >= TARGET_PER_CLASS and count_neg >= TARGET_PER_CLASS:
                break
            
            # Get review data - Amazon Reviews 2023 uses 'rating' and 'text' fields
            stars = sample.get("rating", sample.get("stars", sample.get("star_rating")))
            review_body = sample.get("text", sample.get("review_body", ""))
            review_id = sample.get("user_id", sample.get("review_id", len(processed_reviews)))
            review_date = sample.get("timestamp", sample.get("review_date", sample.get("date")))
            
            # Filter: Keep only 1 or 5 star reviews
            if stars not in TARGET_STARS:
                continue
            
            # Skip empty reviews
            if not review_body or not review_body.strip():
                continue
            
            # Convert stars to binary label: 5 -> 1 (Positive), 1 -> 0 (Negative)
            label = 1 if stars == 5.0 else 0
            
            # Check class-specific limits
            if label == 1 and count_pos >= TARGET_PER_CLASS:
                continue  # Skip, already have enough positives
            if label == 0 and count_neg >= TARGET_PER_CLASS:
                continue  # Skip, already have enough negatives
            
            processed_reviews.append({
                "text": review_body,
                "label": label,
                "review_id": review_id,
                "review_date": review_date
            })
            
            # Update counters
            if label == 1:
                count_pos += 1
            else:
                count_neg += 1
            
            pbar.update(1)
    
    print(f"\nCollected: {count_pos} Positives, {count_neg} Negatives")
    
    # Create DataFrame
    df = pd.DataFrame(processed_reviews)
    
    # Convert review_date to datetime objects
    # Amazon Reviews 2023 uses Unix timestamps (milliseconds)
    df["review_date"] = pd.to_datetime(df["review_date"], unit="ms", errors="coerce")
    
    # Sort by date (ascending) - crucial for drift plot
    df = df.sort_values(by="review_date", ascending=True).reset_index(drop=True)
    
    # Print date range
    min_date = df["review_date"].min()
    max_date = df["review_date"].max()
    print(f"Data sorted by date. Range: {min_date} to {max_date}")
    
    return df


def save_and_verify(df: pd.DataFrame, output_path: str) -> None:
    """Save DataFrame to pickle and print verification info."""
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to pickle
    df.to_pickle(output_path)
    print(f"\nDataFrame saved to: {output_path}")
    
    # Verification: Print first 3 rows
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    print("\nFirst 3 rows:")
    print(df.head(3).to_string())
    
    # Class balance
    total_samples = len(df)
    positive_count = df["label"].sum()
    negative_count = total_samples - positive_count
    positive_pct = (positive_count / total_samples) * 100 if total_samples > 0 else 0
    negative_pct = (negative_count / total_samples) * 100 if total_samples > 0 else 0
    
    print(f"\nClass Balance:")
    print(f"  Total samples: {total_samples}")
    print(f"  Positive (5 stars → 1): {positive_count} ({positive_pct:.2f}%)")
    print(f"  Negative (1 star → 0):  {negative_count} ({negative_pct:.2f}%)")


def main():
    """Main execution function."""
    # Load and process reviews
    df = load_amazon_electronics_reviews()
    
    # Save and verify
    save_and_verify(df, OUTPUT_PATH)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
