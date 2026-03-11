from gen.messages_pb2 import UpsertRequest, UpsertResult
from nodes.pinecone_writer import pinecone_writer


class _NoOpLogger:
    """Minimal AxiomLogger implementation for unit tests."""
    def debug(self, msg: str, **attrs) -> None: pass
    def info(self, msg: str, **attrs) -> None: pass
    def warn(self, msg: str, **attrs) -> None: pass
    def error(self, msg: str, **attrs) -> None: pass


class _NoOpSecrets:
    def get(self, name: str):
        return "", False


def test_pinecone_writer_missing_secret():
    """Without secrets, the node should return upserted_count=0."""
    log = _NoOpLogger()
    secrets = _NoOpSecrets()
    req = UpsertRequest(vector=[0.1, 0.2, 0.3], id="vec-1", text="Hello Axiom")
    result = pinecone_writer(log, secrets, req)
    assert isinstance(result, UpsertResult)
    assert result.upserted_count == 0
