"""
Functions to create final RAG chain to do RAG-Fusion.
"""

from operator import itemgetter

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from treatment_checker.utils import read_text


def create_rag_chain(
    llm: ChatOpenAI, template_path: str, rrf_chain: Runnable
) -> Runnable:
    template = read_text(template_path)
    rag_prompt = ChatPromptTemplate.from_template(template)
    rrf = {"context": rrf_chain, "question": itemgetter("question")}
    parser = StrOutputParser()
    rag_chain = rrf | rag_prompt | llm | parser
    return rag_chain
