import os

import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


def build_vectorstore(df: pd.DataFrame, persist_path: str = "vectorstore") -> FAISS:
    load_dotenv()

    # CRITICAL: Fail early if API key is missing
    if not os.getenv("OPENAI_API_KEY"):
        raise OSError("ERROR: OPENAI_API_KEY environment variable is not set.")

    if "text" not in df.columns:
        raise ValueError("ERROR: Your DataFrame must contain a column named 'text'.")

    texts = df["text"].astype(str).tolist()
    # Ensure all metadata columns exist before converting to dict
    metadata_cols = ["company", "cik", "date"]
    for col in metadata_cols:
        if col not in df.columns:
            # Setting a default empty string for missing metadata columns
            df[col] = ""

    # Filter columns and prepare metadata
    metadatas = df[metadata_cols].fillna("").to_dict(orient="records")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    try:
        test = embeddings.embed_query("hello")
        print(f"Embedding test ok: vector dimension is {len(test)}")
    except Exception as e:
        raise RuntimeError(
            f"Failed to test OpenAI embeddings. Check API key/model access. Error: {e}"
        )

    vectorstore = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)

    vectorstore.save_local(persist_path)
    print("Vectorstore saved at:", persist_path)

    return vectorstore
