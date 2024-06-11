"""
Main script to build the databases required by the project.

Currently it uses Chroma to create simple local vector stores.
"""

from dotenv import load_dotenv
from langchain_openai.embeddings import OpenAIEmbeddings
from loguru import logger

from treatment_checker.build_db.drugs import (
    scrape_drug_data,
    split_drug_data_docs,
)
from treatment_checker.build_db.patient_hist import patient_hist_to_docs
from treatment_checker.build_db.vec_store import docs_to_vectorstore
from treatment_checker.utils import read_json


def build_db():
    # Set up logging and configuration.
    load_dotenv()
    logfile = "data/logs/build_db.log"
    logger.add(logfile, colorize=True, enqueue=True, mode="w")
    db_cfg = read_json("treatment_checker/build_db/db_cfg.json")
    logger.info("Logs initialised.")

    # Get drugs data.
    logger.info("Acquiring drugs data.")
    drug_data_docs = scrape_drug_data(db_cfg["drugs"]["urls"])
    split_drugs_docs = split_drug_data_docs(
        docs=drug_data_docs,
        size=db_cfg["drugs"]["db"]["chunk_size"],
        overlap=db_cfg["drugs"]["db"]["chunk_overlap"],
    )

    # Get patients history data.
    patient_hist_docs = patient_hist_to_docs(
        db_cfg["patient_hist"]["paths"], db_cfg["embedding_model"]["context_window"]
    )

    # Load an embedding model for vector store creation.
    logger.info(f"Loading embedding model {db_cfg["embedding_model"]["name"]}.")
    emb_mdl = OpenAIEmbeddings(model=db_cfg["embedding_model"]["name"])

    # Create the drugs data vector store.
    logger.info("Creating drugs vector store.")
    docs_to_vectorstore(
        docs=split_drugs_docs,
        emb_mdl=emb_mdl,
        db_path=db_cfg["drugs"]["db"]["dir"],
        name=db_cfg["drugs"]["db"]["name"],
    )

    # Create the drugs data vector store.
    logger.info("Creating patients history vector store.")
    docs_to_vectorstore(
        docs=patient_hist_docs,
        emb_mdl=emb_mdl,
        db_path=db_cfg["patient_hist"]["db"]["dir"],
        name=db_cfg["patient_hist"]["db"]["name"],
    )
    logger.info("Building DBs complete.")
    return


if __name__ == "__main__":
    build_db()
