"""
Contains functions for making the final choice for routing of the query.
"""

from typing import Any

from langchain_core.runnables import Runnable
from loguru import logger


def choose_route(res: Any) -> str:
    """
    Logic to choose the most relevent chain to the question.
    """
    # Choose the route the query will take.
    res = res.datasource.lower()
    if "hist_chain" in res:
        logger.info("Returning history chain.")
        return "hist_chain"
    if "diag_chain" in res:
        logger.info("Returning diagnosis chain.")
        return "diag_chain"
    if "full_chain" in res:
        logger.info("Returning full chain.")
        return "full_chain"

    # If a unexpected choice is made, raise an error.
    allowed_choices = ["hist_chain", "diag_chain", "full_chain"]
    err_msg = f"Expected route to be one of {allowed_choices} but got {res}."
    logger.exception(err_msg)
    raise ValueError(err_msg)


def apply_choice(
    choice: str, full_chain: Runnable, diag_chain: Runnable, hist_chain: Runnable
) -> Runnable:
    """
    Logic to return the most relevent chain to the question.
    """
    if "hist_chain" in choice:
        logger.info("Returning history chain.")
        return hist_chain
    if "diag_chain" in choice:
        logger.info("Returning diagnosis chain.")
        return diag_chain
    logger.info("Returning full chain.")
    return full_chain  # Default to last remaining option.
