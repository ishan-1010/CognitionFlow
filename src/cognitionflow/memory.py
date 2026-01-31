"""
Persistent vector memory using ChromaDB and local embeddings.
"""
from autogen.agentchat.contrib.capabilities.teachability import Teachability
from chromadb.utils import embedding_functions


class ZeroCostMemory(Teachability):
    """
    Overriding the default memory class to use HuggingFace embeddings.
    This ensures the memory layer is 100% free and runs locally.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

    def get_embedding_function(self):
        return self.ef


def create_memory(
    path_to_db_dir: str = "./tmp/agent_memory_db",
    reset_db: bool = False,
    recall_threshold: float = 1.5,
) -> ZeroCostMemory:
    """Create and return a ZeroCostMemory instance."""
    return ZeroCostMemory(
        reset_db=reset_db,
        path_to_db_dir=path_to_db_dir,
        recall_threshold=recall_threshold,
    )
