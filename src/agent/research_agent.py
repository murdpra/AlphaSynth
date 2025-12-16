import os
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import SecretStr


class ResearchAgent:
    def __init__(self, index_path: str = "vectorstore") -> None:
        # -----------------------------------------------
        # 1. ROBUST PATH RESOLUTION (CRITICAL FIX: Targeting src/rag/vectorstore)
        # -----------------------------------------------
        # Project structure assumption: [Project Root]/src/agent/research_agent.py
        agent_dir = Path(__file__).resolve().parent  # /path/to/AlphaSynth/src/agent
        src_dir = agent_dir.parent  # /path/to/AlphaSynth/src

        # 💡 FIX: The path must go to the 'rag' subdirectory within 'src'
        rag_dir = src_dir / "rag"

        # The final path should be /path/to/AlphaSynth/src/rag/vectorstore
        final_index_path = rag_dir / index_path
        final_index_path_str = str(final_index_path)

        # -----------------------------------------------
        # 2. LLM and EMBEDDINGS SETUP
        # -----------------------------------------------
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise OSError("OPENAI_API_KEY environment variable is not set.")
        secret_api_key = SecretStr(api_key)

        self.llm_analyst = ChatOpenAI(model="gpt-4o", api_key=secret_api_key)
        self.llm_summarizer = ChatOpenAI(model="gpt-3.5-turbo", api_key=secret_api_key)
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small", openai_api_key=secret_api_key
        )

        # -----------------------------------------------
        # 3. VECTORSTORE LOADING
        # -----------------------------------------------
        try:
            self.vectorstore = FAISS.load_local(
                final_index_path_str,
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True,
            )
        except Exception as e:
            print(
                f"CRITICAL: Failed to load FAISS vector store from {final_index_path_str}. \
                Error: {e}"
            )
            print(
                "\n--------------------------------------------------------------------------------"
            )
            print(f"🛑 ACTION REQUIRED: Vector store not found at {final_index_path_str}")
            print("Please run 'python setup_data.py' to create or move the RAG index.")
            print(
                "--------------------------------------------------------------------------------\n"
            )
            raise FileNotFoundError(f"FAISS index not found or corrupt at {final_index_path_str}")

    # ... (rest of the methods: retrieve_documents, summarize_chunk, analyze) ...
    # (These methods remain the same as the previous full rewrite)

    def retrieve_documents(self, query: str, k: int) -> list[Document]:
        """Retrieve top-k most relevant 10-K chunks."""
        return self.vectorstore.similarity_search(query, k=k)

    def summarize_chunk(self, chunk_text: str) -> str:
        """
        Summarize a long 10-K chunk by splitting it into smaller sub-chunks
        and summarizing them individually for better context fitting.
        """

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, chunk_overlap=100, separators=["\n\n", "\n", " ", ""]
        )
        sub_chunks = text_splitter.split_text(chunk_text)

        sub_chunks = sub_chunks[:3]

        summaries: list[str] = []
        for i, sub in enumerate(sub_chunks):
            prompt = f"Summarize the key points of this 10-K section (Part {i + 1} \
            of {len(sub_chunks)}) concisely for a financial analyst:\n\n{sub}"
            try:
                summary = str(self.llm_summarizer.invoke([HumanMessage(content=prompt)]).content)
                summaries.append(summary)
            except Exception as e:
                print(f"Warning: Summarization failed for a sub-chunk. Error: {e}")
                summaries.append(f"Summarization Failed: {str(e)}")

        return " ".join(summaries)

    def analyze(self, query: str, k: int = 4) -> str:
        """Run retrieval + LLM reasoning with summarization."""
        results = self.retrieve_documents(query, k=k)

        summarized_texts: list[str] = []
        for r in results:
            company = r.metadata.get("company", "UNKNOWN")
            date = r.metadata.get("date", "N/A")
            summary = self.summarize_chunk(r.page_content)
            summarized_texts.append(f"[Filing: {company} - {date}] {summary}")

        doc_text = "\n\n---\n\n".join(summarized_texts)

        prompt = f"""
You are a senior financial research analyst. Analyze the following summarized 10-K filing sections
and answer the query in a highly structured, bulleted manner.

Query:
{query}

Summarized 10-K Sections:
{doc_text}

Provide:
* **Key Risks**: (3-5 concise bullet points based on the filings)
* **Business Overview/Strategy**: (2-3 concise bullet points)
* **Red Flags/Concerns**: (Identify specific, concerning statements)
* **Important Numbers or Trends**: (Reference quantifiable data or strategic trends)
"""
        try:
            return str(self.llm_analyst.invoke([HumanMessage(content=prompt)]).content)
        except Exception as e:
            return f"Final analysis failed: {e}. Raw data summarized:\n\n{doc_text}"
