"""
Contains functions for making the final choice for routing of the query.
"""

from loguru import logger


def choose_route(res):
    """
    Logic to choose the route the query will take.
    """
    res = res.datasource.lower()
    if "patient_history_db" in res:
        logger.info("Setting retriever to patient_history_db.")
        return "set retriever for patient_history_db"
    if "drug_db" in res:
        logger.info("Setting retriever to drug_db.")
        return "set retriever for drug_db"
    err_msg = f"Expected route to be patient_history_db or drug_db but got {res}."
    logger.exception(err_msg)
    raise ValueError(err_msg)
