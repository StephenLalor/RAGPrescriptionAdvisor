"""
Main script to build the databases required by the project.
"""

from loguru import logger

from treatment_checker.build_db.build_drug_db import (
    docs_to_vectorstore,
    scrape_drug_data,
    split_drug_data_docs,
)
from treatment_checker.utils import read_json


def build_db():
    # Set up logging and configuration.
    logfile = "data/logs/build_db.log"
    logger.add(logfile, colorize=True, enqueue=True, mode="w")
    db_cfg = read_json("treatment_checker/build_db/db_cfg.json")
    logger.info("Logs initialised.")

    # Get data.
    logger.info("Acquiring drugs data.")
    drug_data_docs = scrape_drug_data(db_cfg["drug_info_urls"])
    split_docs = split_drug_data_docs(
        docs=drug_data_docs,
        size=db_cfg["drugs_db_chunk_size"],
        overlap=db_cfg["drugs_db_chunk_overlap"],
    )

    # Create vector store.
    logger.info("Creating drugs vector store.")
    docs_to_vectorstore(
        docs=split_docs,
        open_ai_mdl=db_cfg["embedding_model"],
        db_path=db_cfg["drug_db_dir"],
        name=db_cfg["drugs_db_name"],
    )
    logger.info("Building DBs complete.")
    return


if __name__ == "__main__":
    build_db()
