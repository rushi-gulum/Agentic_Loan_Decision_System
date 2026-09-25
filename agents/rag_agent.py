"""
agents/rag_agent.py
===================
RAG-based RBI Guideline Retrieval Agent

Cloud Routing Logic:
  1. CHROMA_CLOUD_API_KEY + CHROMA_CLOUD_TENANT set  →  Chroma Cloud (managed)
  2. Otherwise                                        →  PersistentClient (local ./data/embeddings)

Both paths expose the same `retrieve_feature_guidelines()` interface so the
rest of the system never needs to know which backend is active.
"""

import os
import re
import logging
import warnings
from typing import Optional

warnings.filterwarnings("ignore", category=DeprecationWarning)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_NAME       = "sentence-transformers/all-MiniLM-L6-v2"
LOCAL_CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/embeddings")
COLLECTION_NAME  = os.getenv("CHROMA_COLLECTION_NAME", "rbi_guidelines")

# ---------------------------------------------------------------------------
# 1.  Chroma client factory — Cloud or Local
# ---------------------------------------------------------------------------

def get_chroma_client():
    """
    Return a chromadb client routed to either:
      • Chroma Cloud  (if CHROMA_CLOUD_API_KEY + CHROMA_CLOUD_TENANT are set)
      • Local disk    (fallback, always works in dev)

    NOTE: chromadb.CloudClient was introduced in chroma 0.5+.
    We guard the import so the app degrades gracefully on older installs.
    """
    api_key = os.getenv("CHROMA_CLOUD_API_KEY", "")
    tenant  = os.getenv("CHROMA_CLOUD_TENANT", "")
    database = os.getenv("CHROMA_CLOUD_DATABASE", "rbi_guidelines")

    if api_key and tenant:
        try:
            import chromadb
            client = chromadb.HttpClient(
                host="api.trychroma.com",
                ssl=True,
                headers={
                    "x-chroma-token": api_key,
                    "X-Chroma-Tenant": tenant,
                    "X-Chroma-Database": database,
                },
            )
            # ping to confirm connectivity
            client.heartbeat()
            logger.info("☁️  Connected to Chroma Cloud (tenant=%s, db=%s)", tenant, database)
            return client, "cloud"
        except Exception as exc:
            logger.warning("⚠️  Chroma Cloud unavailable (%s) — falling back to local", exc)

    # Local persistent fallback
    import chromadb
    os.makedirs(LOCAL_CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=LOCAL_CHROMA_DIR)
    logger.info("💾 Connected to local Chroma at %s", LOCAL_CHROMA_DIR)
    return client, "local"


def get_vector_store():
    """
    Return the ChromaDB collection for RBI guidelines.
    Works transparently with both cloud and local backends.
    """
    client, mode = get_chroma_client()
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    logger.debug("📚 Collection '%s' ready (%s mode, %d docs)",
                 COLLECTION_NAME, mode, collection.count())
    return collection


# ---------------------------------------------------------------------------
# 2.  Embedding helper (HuggingFace, local model — no API cost)
# ---------------------------------------------------------------------------

def _get_embedder():
    """Lazy-load the sentence-transformer embedder."""
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception as exc:
        logger.error("Embedder init failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# 3.  LangChain Chroma wrapper (for similarity_search convenience)
# ---------------------------------------------------------------------------

def _get_langchain_vectorstore():
    """
    Wraps the raw Chroma collection in a LangChain Chroma object so we
    can call .similarity_search() the same way in both cloud and local mode.
    """
    embedder = _get_embedder()
    if embedder is None:
        return None

    api_key = os.getenv("CHROMA_CLOUD_API_KEY", "")
    tenant  = os.getenv("CHROMA_CLOUD_TENANT", "")
    database = os.getenv("CHROMA_CLOUD_DATABASE", "rbi_guidelines")

    if api_key and tenant:
        try:
            import chromadb
            from langchain_chroma import Chroma as LCChroma

            http_client = chromadb.HttpClient(
                host="api.trychroma.com",
                ssl=True,
                headers={
                    "x-chroma-token": api_key,
                    "X-Chroma-Tenant": tenant,
                    "X-Chroma-Database": database,
                },
            )
            return LCChroma(
                client=http_client,
                collection_name=COLLECTION_NAME,
                embedding_function=embedder,
            )
        except Exception as exc:
            logger.warning("⚠️  LangChain Chroma Cloud failed (%s) — using local", exc)

    # Local fallback
    from langchain_chroma import Chroma as LCChroma
    return LCChroma(
        persist_directory=LOCAL_CHROMA_DIR,
        collection_name=COLLECTION_NAME,
        embedding_function=embedder,
    )


# ---------------------------------------------------------------------------
# 4.  Text summarisation helpers
# ---------------------------------------------------------------------------

def clean_guideline_text(text: str) -> str:
    """Light cleaning — remove boilerplate, collapse whitespace."""
    if not text or len(text.strip()) < 30:
        return "No direct RBI rule found."
    text = re.sub(
        r"(no direct rbi rule|unable to find|not mentioned|no explicit mention).*?\n",
        "", text, flags=re.I,
    )
    text = re.sub(r"\s+", " ", text).strip()[:2000]
    return text


def summarize_guidelines(raw_texts: list) -> str:
    """Use the LLM to distil retrieved RBI text into bullet-point rules."""
    if not raw_texts:
        return "No direct RBI rule found."

    cleaned = clean_guideline_text(" ".join(raw_texts))
    if "No direct RBI rule found" in cleaned:
        return "No direct RBI rule found."

    try:
        from utils.llm_utility import get_llm
        from langchain.prompts import PromptTemplate
        from langchain.chains import LLMChain

        llm = get_llm()
        prompt = PromptTemplate.from_template("""
You are an RBI policy expert.
Extract specific *numeric* or *rule-based* limits from the text below.
Examples: "LTV ≤ 80%", "FOIR ≤ 50%", "Age ≥ 21 years".
Give 1–3 bullet points. If nothing specific, reply: "No direct RBI rule found."

RBI Text:
{text}

Summary:
""")
        chain = LLMChain(llm=llm, prompt=prompt)
        summary = chain.run({"text": cleaned}).strip()

        if re.search(r"paste the text|unable to access", summary, flags=re.I):
            return "No direct RBI rule found."

        lines = [ln.strip("•* ") for ln in summary.splitlines() if ln.strip()]
        return " ".join(lines[:3]) or "No direct RBI rule found."

    except Exception as exc:
        logger.warning("LLM summarisation failed: %s", exc)
        return cleaned[:300]


# ---------------------------------------------------------------------------
# 5.  Public interface — used by compliance_agent and orchestrator
# ---------------------------------------------------------------------------

RELEVANT_FEATURES = [
    "age_years", "monthly_income_inr", "foir_total_obligations_pct",
    "ltv_ratio", "pep_flag", "interest_type", "loan_type",
]


def retrieve_feature_guidelines(
    applicant_data: dict,
    top_k: int = 3,
) -> dict:
    """
    For each relevant applicant feature, retrieve RBI guideline snippets
    from ChromaDB (cloud or local) and return concise policy summaries.

    Args:
        applicant_data: loan application dict
        top_k: number of chunks to retrieve per feature

    Returns:
        {
          "loan_type": "housing",
          "chroma_mode": "cloud" | "local",
          "feature_guidelines": {
              "bureau_score": {"value": 720, "summary": "..."},
              ...
          }
        }
    """
    lc_store = _get_langchain_vectorstore()
    if lc_store is None:
        logger.warning("Vector store unavailable — returning empty guidelines")
        return {"loan_type": applicant_data.get("loan_type"), "feature_guidelines": {}}

    # Detect which backend is active for telemetry
    chroma_mode = "cloud" if (
        os.getenv("CHROMA_CLOUD_API_KEY") and os.getenv("CHROMA_CLOUD_TENANT")
    ) else "local"

    feature_guidelines: dict = {}

    for feature in RELEVANT_FEATURES:
        value = applicant_data.get(feature)
        if value is None:
            continue

        clean_feature = re.sub(r"_", " ", feature)
        query = (
            f"RBI guideline related to {clean_feature} "
            f"{value} for {applicant_data.get('loan_type', 'loan')}"
        )

        try:
            retrieved_docs = lc_store.similarity_search(query, k=top_k)
            retrieved_texts = [d.page_content for d in retrieved_docs]
        except Exception as exc:
            logger.warning("Retrieval failed for feature '%s': %s", feature, exc)
            retrieved_texts = []

        summary = summarize_guidelines(retrieved_texts)

        feature_guidelines[feature] = {
            "value": value,
            "query": query,
            "retrieved_guidelines": retrieved_texts[:1],
            "summary": summary,
        }

    return {
        "loan_type": applicant_data.get("loan_type"),
        "chroma_mode": chroma_mode,
        "feature_guidelines": feature_guidelines,
    }


# ---------------------------------------------------------------------------
# 6.  Health-check used by api/app.py /health endpoint
# ---------------------------------------------------------------------------

def check_rag_health() -> dict:
    """Returns vector-store connectivity status for monitoring."""
    try:
        client, mode = get_chroma_client()
        collection = client.get_or_create_collection(COLLECTION_NAME)
        count = collection.count()
        return {
            "status": "healthy",
            "mode": mode,
            "collection": COLLECTION_NAME,
            "document_count": count,
        }
    except Exception as exc:
        return {
            "status": "unhealthy",
            "mode": "unknown",
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("🩺 RAG health:", check_rag_health())

    sample = {
        "loan_type": "housing",
        "age_years": 32,
        "monthly_income_inr": 55000,
        "foir_total_obligations_pct": 45.0,
        "ltv_ratio": 0.75,
        "pep_flag": False,
    }
    result = retrieve_feature_guidelines(sample, top_k=2)
    print("📋 Guidelines retrieved for", len(result["feature_guidelines"]), "features")
    print("   Chroma mode:", result["chroma_mode"])
