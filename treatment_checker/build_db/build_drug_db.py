"""
Functions for scraping, parsing and embedding drugs data to a vector store.
"""

from bs4 import SoupStrainer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from loguru import logger


def scrape_drug_data(urls: list[str]) -> list[Document]:
    """
    Scrape drug data from a list of drugs.com URLs and parse into langchain docs.
    """
    logger.info(f"Scraping {len(urls)} URLs: {urls}.")
    classes_to_parse = "ddc-main-content"  # Only need to parse main content.
    parsing_settings = {"parse_only": SoupStrainer(class_=(classes_to_parse))}
    doc_loader = WebBaseLoader(web_paths=urls, bs_kwargs=parsing_settings)
    docs = doc_loader.load()  # Actually scrape content.
    return docs


def split_drug_data_docs(
    docs: list[Document], size: int, overlap: int
) -> list[Document]:
    """
    Split docs using recursive character splitter, based on token count.
    """
    logger.info(f"Splitting {len(docs)}.")
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=size, chunk_overlap=overlap
    )
    split_docs = text_splitter.split_documents(docs)  # Splits of chunk_size tokens.
    logger.info(f"Split into {len(split_docs)} docs.")
    return split_docs


def docs_to_vectorstore(
    docs: list[Document], open_ai_mdl: str, db_path: str, name: str
) -> Chroma:
    """
    Create local vector store using Chroma with docs and their embeddings.
    """
    logger.info(f"Creating Chroma store with docs and embeddings using {open_ai_mdl}.")
    embedding_mdl = OpenAIEmbeddings(model=open_ai_mdl)
    Chroma.from_documents(
        documents=docs,
        embedding=embedding_mdl,
        persist_directory=db_path,
        collection_name=name,
    )
    logger.info(f"Chroma vector store {name} created in {db_path}.")
    return
