"""
Functions to implement reciprocal rank fusion for rag fusion.
"""

from collections import defaultdict

from langchain.load import dumps, loads
from langchain_core.documents import Document
from langchain_core.runnables import Runnable
from langchain_core.vectorstores import VectorStoreRetriever
from loguru import logger


def get_unique_union(docs: list[list]) -> list[Document]:
    """
    Take the unique union of all retrieved docs to avoid duplicated LLM input.
    """
    # For all returned documents, cast them to JSON so we can use set to dedupe.
    flat_json_docs = [dumps(doc) for sublist in docs for doc in sublist]
    logger.info(f"Getting unique union of {len(flat_json_docs)} docs.")
    unq_json_docs = list(set(flat_json_docs))
    logger.info(f"Returning {len(unq_json_docs)} docs.")

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
    logger.info(f"Calculating RRF scores for {len(res)} docs.")
    fused_scores = defaultdict(float)  # Initial score defaults to zero.
    for docs in res:
        for rank, doc in enumerate(docs):  # Doc rank is position in list of retrieves.
            doc_as_str = dumps(doc)  # Dict requires a hashable key.
            fused_scores[doc_as_str] += 1 / (rank + k)  # Apply formula to get score.

    # Sort docs based on their fused scores and return re-ranked docs.
    sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    reranked_results = [(loads(doc), score) for doc, score in sorted_docs]
    logger.info(f"Returning {len(reranked_results)} re-ranked docs.")

    return reranked_results


def create_rrf_chain(multi_q_chain: Runnable, ret: VectorStoreRetriever) -> Runnable:
    """
    Reciprocal ranked fusion.

    Retrieval based on multiple queries, then dedupe and rank the results.
    """
    logger.info("Creating RRF chain.")
    rrf_chain = multi_q_chain | ret.map() | get_unique_union | reciprocal_rank_fusion
    return rrf_chain
