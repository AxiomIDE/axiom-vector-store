from gen.messages_pb2 import QueryRequest, RetrievalResult
from gen.axiom_logger import AxiomLogger, AxiomSecrets


def pinecone_retriever(log: AxiomLogger, secrets: AxiomSecrets, input: QueryRequest) -> RetrievalResult:
    """Performs a top-k nearest-neighbour search in a Pinecone index and returns the matching text chunks.

    Reads PINECONE_API_KEY and PINECONE_INDEX from secrets. Returns the text
    stored in the "text" metadata field of each matching vector, ordered by
    similarity score descending. Defaults to top_k=5 when input.top_k is zero.
    """
    from pinecone import Pinecone

    api_key, ok = secrets.get("PINECONE_API_KEY")
    if not ok:
        log.error("pinecone_retriever: PINECONE_API_KEY secret not found")
        return RetrievalResult(chunks=[], scores=[])

    index_name, ok = secrets.get("PINECONE_INDEX")
    if not ok:
        log.error("pinecone_retriever: PINECONE_INDEX secret not found")
        return RetrievalResult(chunks=[], scores=[])

    top_k = input.top_k if input.top_k > 0 else 5

    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)

    log.info("pinecone_retriever: querying", dim=len(input.vector), top_k=top_k, index=index_name)
    response = index.query(vector=list(input.vector), top_k=top_k, include_metadata=True)

    chunks = []
    scores = []
    for match in response.matches:
        text = match.metadata.get("text", "") if match.metadata else ""
        chunks.append(text)
        scores.append(match.score)

    log.info("pinecone_retriever: retrieved", matches=len(chunks))
    return RetrievalResult(chunks=chunks, scores=scores, question=input.question)
