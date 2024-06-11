from operator import itemgetter

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import ChatOpenAI
from loguru import logger

from treatment_checker.rag.multi_query import create_multi_query_chain
from treatment_checker.rag.reciprocal_rank_fusion import create_rrf_chain
from treatment_checker.utils import read_text


def get_full_chain(
    llm: ChatOpenAI, hist_ret: VectorStoreRetriever, drugs_ret: VectorStoreRetriever
) -> Runnable:
    """
    This chain does RAG using both vector stores.
    """
    # Set up patient identification chain.
    logger.info("Creating patient ID prompt from original question.")
    template = read_text("treatment_checker/rag/data/identify_patient_template.txt")
    id_prompt = ChatPromptTemplate.from_template(template)
    parser = StrOutputParser()
    patient_id_chain = id_prompt | llm | parser | hist_ret

    # Set up chains for RAG-Fusion.
    logger.info("Creating multiquery chain.")
    multi_q_chain = create_multi_query_chain(llm)
    logger.info("Creating RRF chain.")
    rrf_chain = create_rrf_chain(multi_q_chain, drugs_ret)
    rrf_chain_context = {
        "drug_info": rrf_chain,
        "patient_hist": patient_id_chain,
        "question": itemgetter("question"),
    }

    # Set up final prompt and combine.
    logger.info("Creating final RAG prompt.")
    template = read_text("treatment_checker/rag/data/rag_with_hist_template.txt")
    rag_prompt = ChatPromptTemplate.from_template(template)

    return rrf_chain_context | rag_prompt | llm | parser
