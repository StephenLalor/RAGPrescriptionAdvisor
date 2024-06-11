"""
Contains functions for reading and parsing patient history files.

Simple approach for now but we could add more sophisticated chunking etc later.
"""

import tiktoken
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from loguru import logger


def patient_hist_to_docs(paths: list[str], context_window: int) -> list[Document]:
    """
    Read patient history files and parse into langchain docs, checking token length is not too large.

    Using token length check here as we aren't explicitly chunking, and are reading whole files.
    """
    # Load and parse the text files to docs.
    logger.info(f"Loading {len(paths)} patient history files.")
    docs = [TextLoader(path).load()[0] for path in paths]

    # Check the tokens length of the result is not too large.
    for size in est_tokens_in_docs(docs):
        try:
            if size > context_window:
                raise TooManyTokensError(size, context_window)
        except TooManyTokensError as e:
            logger.exception(e)

    # Return docs if safe.
    return docs


def est_tokens_in_docs(docs: list[Document]) -> list[int]:
    """
    Estimate the number of tokens for each doc in the list.

    This is just an estimate because cl100k_base may not match the tokenisation used by OpenAI's embedding models.
    """
    encoding = tiktoken.get_encoding("cl100k_base")  # Guess as to best encoding to use!
    return [len(encoding.encode(doc.page_content)) for doc in docs]


class TooManyTokensError(Exception):
    def __init__(self, num_tokens, limit):
        self.num_tokens = num_tokens
        self.limit = limit
        self.message = f"Chunk contains too many tokens ({num_tokens} > {limit})"
        super().__init__(self.message)
