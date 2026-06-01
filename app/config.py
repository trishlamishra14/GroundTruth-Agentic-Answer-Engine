"""
app/config.py — every setting in one typed place, loaded from .env.

Import the single `settings` object anywhere: `from app.config import settings`.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- Cohere ---
    # Default "" lets modules import without a key (e.g. unit tests / CI);
    # any real Cohere call still needs a valid key in .env.
    cohere_api_key: str = ""
    embedding_model: str = "embed-english-v3.0"
    chat_model: str = "command-r-08-2024"
    rerank_model: str = "rerank-v3.5"
    embedding_dim: int = 1024

    # --- Database ---
    database_url: str = "postgresql://rag:rag@localhost:5432/ragdb"

    # --- Retrieval ---
    top_k: int = 8            # candidates pulled from each retriever
    rerank_top_n: int = 4     # passages kept after reranking

    # --- Agent / abstention ---
    abstain_threshold: float = 0.30   # data-calibrated: midpoint of a clean gap
    #                                   (out-of-scope max ~0.10 vs in-scope min ~0.50);
    #                                   see `python -m eval.calibrate`
    max_agent_loops: int = 2          # how many times the agent may rewrite + retry
    use_langgraph: bool = False       # True -> run the LangGraph version of the agent

    # --- Cost estimate (USD per 1M tokens; rough, for the dashboard only) ---
    price_in_per_mtok: float = 0.15
    price_out_per_mtok: float = 0.60

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
