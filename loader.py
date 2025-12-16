import logging

import pandas as pd
from datasets import load_dataset

logger = logging.getLogger(__name__)


def load_research_dataset(
    sample: int | None = None, filter_company: str | None = None
) -> pd.DataFrame:
    """
    Loads 10-K filing data from the Hugging Face dataset, cleans it, and combines
    relevant text columns into a single 'text' column for the vector store.
    """
    logging.info("Attempting to load 10-K dataset from Hugging Face...")
    try:
        ds = load_dataset("jlohding/sp500-edgar-10k", split="train")
        df = pd.DataFrame(ds)
    except Exception as e:
        logging.error(f"Failed to load dataset 'jlohding/sp500-edgar-10k'. Error: {e}")

        raise RuntimeError(f"Data loading failed: {e}")

    # 1. Column Aggregation
    # Identify item columns (Item 1 to Item 15 of a 10-K filing)
    item_cols: list[str] = [col for col in df.columns if col.startswith("item_")]

    if not item_cols:
        logging.warning("No 'item_X' columns found. Data schema may be incorrect.")

    # 💡 Robustly handle missing values (fillna) and combine text fields with clear separator
    df["text"] = df[item_cols].fillna("").astype(str).agg(" \n\n ".join, axis=1)

    # 2. Column Selection and Standardizing Metadata
    required_cols = ["company", "cik", "date", "text"]

    # Ensure only necessary metadata and the 'text' column remain
    df = df.reindex(columns=required_cols)

    # 3. Data Cleaning and Filtering

    # Ensure 'text' column is string type
    df["text"] = df["text"].astype(str)

    # 💡 CRITICAL: Filter out rows where the combined text is too short (empty/corrupted filings)
    df = df[df["text"].str.len() > 100]

    # Drop rows where critical metadata (like 'company' or 'text') is missing
    df = df.dropna(subset=["text", "company"])

    logging.info(f"Dataset loaded and cleaned. Total number of documents: {len(df)}")

    # 4. Sampling and Filtering Logic (Company filter must run first)
    if filter_company:
        initial_count = len(df)
        # Filter by company, case-insensitive
        df = df[df["company"].str.contains(filter_company, case=False, na=False)]
        logging.info(
            f"Filtered by company '{filter_company}'. Retained {len(df)} documents \
                  (Dropped {initial_count - len(df)})."
        )

    if sample and len(df) > sample:
        # Sample only after filtering by company
        df = df.sample(n=sample, random_state=42).reset_index(drop=True)
        logging.info(f"Dataset sampled down to {sample} documents.")

    if df.empty:
        logging.warning("The final DataFrame is empty after filtering/sampling.")

    return df
