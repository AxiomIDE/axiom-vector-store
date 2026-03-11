from gen.messages_pb2 import UpsertRequest, UpsertResult
from gen.axiom_logger import AxiomLogger, AxiomSecrets


def pinecone_writer(log: AxiomLogger, secrets: AxiomSecrets, input: UpsertRequest) -> UpsertResult:
    """Upserts a vector and its source text into a Pinecone index.

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
        return UpsertResult(upserted_count=0)

    index_name, ok = secrets.get("PINECONE_INDEX")
    if not ok:
        log.error("pinecone_writer: PINECONE_INDEX secret not found")
        return UpsertResult(upserted_count=0)

    vector_id = input.id if input.id else str(uuid.uuid4())

    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)

    log.info("pinecone_writer: upserting vector", id=vector_id, dim=len(input.vector), index=index_name)
    index.upsert(vectors=[{"id": vector_id, "values": list(input.vector), "metadata": {"text": input.text}}])
    log.info("pinecone_writer: upsert complete")
    return UpsertResult(upserted_count=1)
