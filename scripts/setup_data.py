# scripts/setup_data.py
import logging
from pathlib import Path

import click

from src.financial_analysis.rag.loader import load_research_dataset
from src.financial_analysis.rag.vector_store import build_vectorstore

# Set up basic logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


@click.command()
@click.option(
    "--data-path",
    default="src/financial_analysis/rag/vectorstore",
    help="The path for the vector store, relative to the project root.",
    type=click.Path(),
)
@click.option(
    "--filter-company",
    default="Microsoft",
    help="The company to filter the dataset by.",
)
@click.option("--sample-size", default=50, help="The number of samples to load.")
def main(data_path, filter_company, sample_size):
    """
    Main function to set up the data and build the vector store.
    """
    # Define project root relative to this script's location
    project_root = Path(__file__).resolve().parents[1]
    target_path = project_root / data_path

    if target_path.exists() and target_path.is_dir() and any(target_path.iterdir()):
        logging.info(
            f"'{data_path}' already exists and contains files. Skipping vector store creation."
        )
    else:
        logging.info("Starting data loading and vector store creation...")
        try:
            # 1. Load and clean the dataset
            logging.info(
                f"Loading data (Sample: {sample_size}, "
                f"Filter: {filter_company if filter_company else 'None'})."
                f"This may take a moment."
            )
            df = load_research_dataset(sample=sample_size, filter_company=filter_company)

            if df.empty:
                logging.error(
                    "DataFrame is empty after loading/filtering. Cannot build vector store."
                )
            else:
                # 2. Build the FAISS vector store
                build_vectorstore(df, persist_path=str(target_path))
                logging.info(
                    f"Vector store successfully created at '{target_path}'! "
                    f"You can now run the API."
                )

        except Exception as e:
            logging.error(
                f"DATA SETUP FAILED. "
                f"Ensure your API key is set and all dependencies are installed."
                f"Error: {e}",
                exc_info=True,
            )


if __name__ == "__main__":
    main()
