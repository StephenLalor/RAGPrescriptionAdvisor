"""
Functions for multi-query query transformation.
"""

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_openai import ChatOpenAI
from loguru import logger

from treatment_checker.utils import read_text, split_on_nl


def create_multi_query_chain(llm: ChatOpenAI) -> Runnable:
    """
    Create a chain to generate multiple queries from the input question.
    """
    # Construct multi-query prompt template.
    logger.info("Creating prompt.")
    template = read_text("treatment_checker/rag/data/multi_query_template.txt")
    prompt = ChatPromptTemplate.from_template(template)

    # Parse output to desired format.
    logger.info("Creating output parsing components.")
    parser = StrOutputParser()  # Only simple string output needed.
    output_splitter = RunnableLambda(split_on_nl)  # Allows func use in chain.

    # Combine all previous steps into chain.
    logger.info("Combining components into chain.")
    gen_multi_query = prompt | llm | parser | output_splitter

    return gen_multi_query
