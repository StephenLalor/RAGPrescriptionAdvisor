from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import ChatOpenAI
from loguru import logger

from treatment_checker.utils import read_text


def get_hist_chain(llm: ChatOpenAI, hist_ret: VectorStoreRetriever) -> Runnable:
    """
    Use standard RAG to generate answers on the patient medical history database.
    """
    logger.info("Running patient history chain.")
    template = read_text("treatment_checker/rag/data/rag_template.txt")
    prompt = ChatPromptTemplate.from_template(template)
    parser = StrOutputParser()
    setup = {"context": hist_ret, "question": RunnablePassthrough()}
    return setup | prompt | llm | parser
