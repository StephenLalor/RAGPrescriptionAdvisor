"""
Functions to implement reciprocal rank fusion for rag fusion.
"""

from collections import defaultdict

from langchain.load import dumps, loads
from langchain_core.documents import Document
from langchain_core.runnables import Runnable
from langchain_core.vectorstores import VectorStoreRetriever


def get_unique_union(docs: list[list]) -> list[Document]:
    """
    Take the unique union of all retrieved docs to avoid duplicated LLM input.
    """
    # For all returned documents, cast them to JSON so we can use set to dedupe.
    flat_json_docs = [dumps(doc) for sublist in docs for doc in sublist]
    unq_json_docs = list(set(flat_json_docs))

    # Cast docs back to langchain documents class.
    unq_docs = [loads(doc) for doc in unq_json_docs]
    return unq_docs


def reciprocal_rank_fusion(res: list[list], k=60) -> list[tuple[Document, float]]:
    """
    Merge the rankings of multiple result sets of retrieved docs to a single set of ranks.
    Using this single set of ranks, re-rank the set of results to get a single set of results and their ranks.

    Note that k is an experimentally determined constant.
    """
    # Assign score using reciprocal rank fusion formula.
    fused_scores = defaultdict(float)  # Initial score defaults to zero.
    for docs in res:
        for rank, doc in enumerate(docs):  # Doc rank is position in list of retrieves.
            doc_as_str = dumps(doc)  # Dict requires a hashable key.
            fused_scores[doc_as_str] += 1 / (rank + k)  # Apply formula to get score.

    # Sort docs based on their fused scores and return re-ranked docs.
    sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    reranked_results = [(loads(doc), score) for doc, score in sorted_docs]

    return reranked_results


def create_rrf_chain(
    gen_multi_query: Runnable, retriever: VectorStoreRetriever
) -> Runnable:
    """
    Create chain to retrieve docs using multi-query, do the retrieval, then dedupe and rank the results.
    """
    rrf_chain = (
        gen_multi_query | retriever.map() | get_unique_union | reciprocal_rank_fusion
    )
    return rrf_chain
