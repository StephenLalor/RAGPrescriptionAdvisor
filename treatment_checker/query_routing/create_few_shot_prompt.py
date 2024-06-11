"""
Contains functions for the creation of a few shot prompt to guide query routing.
"""

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from loguru import logger

from treatment_checker.query_routing.query_router import QueryRouter
from treatment_checker.utils import read_json, read_text


def create_few_shot_prompt() -> FewShotPromptTemplate:
    logger.info("Constructing few shot prompt.")
    parser = PydanticOutputParser(pydantic_object=QueryRouter)
    prompt = FewShotPromptTemplate(
        examples=generate_examples(),
        example_prompt=create_example_prompt(),
        prefix=read_text("treatment_checker/query_routing/data/prefix.txt"),
        suffix=read_text("treatment_checker/query_routing/data/suffix.txt"),
        input_variables=["question"],
        example_separator="\n",
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    return prompt


def generate_examples() -> list[dict]:
    """
    Several simple examples of desired routing responses.
    """
    examples = read_json("treatment_checker/query_routing/data/examples.json")
    return [ex for ex in examples.values()]


def create_example_prompt() -> PromptTemplate:
    """
    Example question-answer pairs for few shot prompt for routing responses.
    """
    examples_template = "[Question]: {question}\n [Answer]: {answer}"
    examples_input = ["question", "answer"]
    return PromptTemplate(template=examples_template, input_variables=examples_input)
