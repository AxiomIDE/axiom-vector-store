from typing import Iterator
from gen.messages_pb2 import UpsertRequest, UpsertResult
from gen.axiom_logger import AxiomLogger, AxiomSecrets


def pinecone_writer(log: AxiomLogger, secrets: AxiomSecrets, inputs: Iterator[UpsertRequest]) -> Iterator[UpsertResult]:
    """Upserts each incoming vector frame into a Pinecone index and streams one UpsertResult per frame.

    Reads PINECONE_API_KEY and PINECONE_INDEX from secrets. Stores the source
    text as metadata under the key "text" so it can be retrieved alongside the
    vector scores. The vector ID is taken from input.id; if empty a UUID is
    generated.
    """
    import uuid
    from pinecone import Pinecone

    api_key, ok = secrets.get("PINECONE_API_KEY")
    if not ok:
        log.error("pinecone_writer: PINECONE_API_KEY secret not found")
        return

    index_name, ok = secrets.get("PINECONE_INDEX")
    if not ok:
        log.error("pinecone_writer: PINECONE_INDEX secret not found")
        return

    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)

    for input in inputs:
        vector_id = input.id if input.id else str(uuid.uuid4())
        log.info("pinecone_writer: upserting vector", id=vector_id, dim=len(input.vector), index=index_name)
        index.upsert(vectors=[{"id": vector_id, "values": list(input.vector), "metadata": {"text": input.text}}])
        log.info("pinecone_writer: upsert complete", id=vector_id)
        yield UpsertResult(upserted_count=1)
