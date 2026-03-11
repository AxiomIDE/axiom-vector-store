from gen.messages_pb2 import QueryRequest, RetrievalResult
from nodes.pinecone_retriever import pinecone_retriever


class _NoOpLogger:
    """Minimal AxiomLogger implementation for unit tests."""
    def debug(self, msg: str, **attrs) -> None: pass
    def info(self, msg: str, **attrs) -> None: pass
    def warn(self, msg: str, **attrs) -> None: pass
    def error(self, msg: str, **attrs) -> None: pass


class _NoOpSecrets:
    def get(self, name: str):
        return "", False


def test_pinecone_retriever_missing_secret():
    """Without secrets, the node should return empty results."""
    log = _NoOpLogger()
    secrets = _NoOpSecrets()
    req = QueryRequest(vector=[0.1, 0.2, 0.3], top_k=5)
    result = pinecone_retriever(log, secrets, req)
    assert isinstance(result, RetrievalResult)
    assert len(result.chunks) == 0
    assert len(result.scores) == 0
