# setup_data.py (Updated to target src/rag/vectorstore)
import logging
import os
import sys
from pathlib import Path  # Import pathlib for robust pathing

# Ensure the current directory is in the Python path for local module imports
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from src.agent.loader import load_research_dataset
from vector import build_vectorstore

# Set up basic logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# --- Configuration ---
# 💡 FIX: Set the path to the correct subdirectory structure
DATA_PATH = "src/rag/vectorstore"
FILTER_COMPANY = "Microsoft"
SAMPLE_SIZE = 50
# --- End Configuration ---

if __name__ == "__main__":
    # Robustly check if the directory exists using the new Path object
    target_path = Path(DATA_PATH)

    if target_path.exists() and target_path.is_dir() and len(os.listdir(target_path)) > 0:
        logging.info(
            f"'{DATA_PATH}' already exists and contains files. Skipping vector store creation."
        )
    else:
        logging.info("Starting data loading and vector store creation...")
        try:
            # 1. Load and clean the dataset
            logging.info(
                f"Loading data (Sample: {SAMPLE_SIZE}, Filter: {
                    FILTER_COMPANY if FILTER_COMPANY else 'None'
                }). This may take a moment."
            )
            df = load_research_dataset(sample=SAMPLE_SIZE, filter_company=FILTER_COMPANY)

            if df.empty:
                logging.error(
                    "DataFrame is empty after loading/filtering. Cannot build vector store."
                )
            else:
                # 2. Build the FAISS vector store
                # This call now saves the data to src/rag/vectorstore
                build_vectorstore(df, persist_path=DATA_PATH)
                logging.info("Vector store successfully created! You can now run app.py.")

        except Exception as e:
            logging.error(
                f"DATA SETUP FAILED. Ensure your API key is set and all dependencies are \
                installed. Error: {e}"
            )
