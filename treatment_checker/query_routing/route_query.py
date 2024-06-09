"""
Contains functions for creating the final query routing chain.
"""

from langchain_core.runnables import Runnable, RunnableLambda
from langchain_openai import ChatOpenAI
from loguru import logger

from treatment_checker.query_routing.choose_route import choose_route
from treatment_checker.query_routing.create_few_shot_prompt import (
    create_few_shot_prompt,
)
from treatment_checker.query_routing.query_router import QueryRouter


def query_routing(llm: ChatOpenAI) -> Runnable:
    logger.info("Routing query.")
    prompt = create_few_shot_prompt()
    struct_llm = llm.with_structured_output(QueryRouter)
    choice = RunnableLambda(choose_route)
    router_chain = prompt | struct_llm | choice
    return router_chain
