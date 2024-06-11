"""
Functions for creating vector stores with documents and their embeddings.
"""

from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings
from loguru import logger


def docs_to_vectorstore(
    docs: list[Document], emb_mdl: OpenAIEmbeddings, db_path: str, name: str
) -> None:
    """
    Create local vector store using Chroma with docs and their embeddings.
    """
    logger.info(f"Creating Chroma store with docs and embeddings using {emb_mdl}.")
    Chroma.from_documents(
        documents=docs,
        embedding=emb_mdl,
        persist_directory=db_path,
        collection_name=name,
    )
    logger.info(f"Chroma vector store {name} created in {db_path}.")
    return


def load_retriever(
    emb_mdl: OpenAIEmbeddings, dir_path: str, name: str, k: int
) -> VectorStoreRetriever:
    """
    Load the vector store and return it as a retriever.
    """
    logger.info(f"Loading {dir} as retriever.")
    vector_store = Chroma(
        persist_directory=dir_path, embedding_function=emb_mdl, collection_name=name
    )
    return vector_store.as_retriever(search_kwargs={"k": k})
