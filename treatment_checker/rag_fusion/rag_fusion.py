"""
Main script to run RAG-Fusion.
"""

from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from loguru import logger

from treatment_checker.rag_fusion.multi_query import create_multi_query_chain
from treatment_checker.rag_fusion.rag import create_rag_chain
from treatment_checker.rag_fusion.reciprocal_rank_fusion import create_rrf_chain
from treatment_checker.utils import read_json


def run_rag(question):
    # Set up logging and configuration.
    logfile = "data/logs/run_rag.log"
    logger.add(logfile, colorize=True, enqueue=True, mode="w")
    rag_cfg = read_json("treatment_checker/rag_fusion/rag_cfg.json")
    db_cfg = read_json("treatment_checker/build_db/db_cfg.json")
    logger.info("Logs initialised.")

    # Load a chat model with completions.
    logger.info(f"Loading {rag_cfg["chat_model"]} (temp: {rag_cfg["temp"]}).")
    llm = ChatOpenAI(model=rag_cfg["chat_model"], temperature=rag_cfg["temp"])

    # Load database and create retriever.
    logger.info(f"Loading vector store ({db_cfg["drug_db_dir"]}).")
    emb_mdl = OpenAIEmbeddings(model=db_cfg["embedding_model"])
    vector_store = Chroma(
        persist_directory=db_cfg["drug_db_dir"], embedding_function=emb_mdl
    )
    retriever = vector_store.as_retriever()

    # Create RAG chain with reciprocal rank fusion.
    logger.info("Assembling RAG-Fusion chain.")
    multi_q = create_multi_query_chain(llm, rag_cfg["mq_template_path"])
    rrf = create_rrf_chain(multi_q, retriever)
    rag = create_rag_chain(llm, rag_cfg["rag_template_path"], rrf)

    # Generate answer using RAG-fusion.
    logger.info("Invoking RAG-Fusion chain.")
    rag.invoke({"question": question})


if __name__ == "__main__":
    run_rag("Why is aspirin usually prescribed?")
