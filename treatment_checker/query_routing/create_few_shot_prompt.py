"""
Contains functions for the creation of a few shot prompt to guide query routing.
"""

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from loguru import logger

from treatment_checker.query_routing.query_router import QueryRouter
from treatment_checker.utils import read_text


def create_few_shot_prompt() -> FewShotPromptTemplate:
    logger.info("Constructing few shot prompt.")
    parser = PydanticOutputParser(pydantic_object=QueryRouter)
    prompt = FewShotPromptTemplate(
        examples=generate_examples(),
        example_prompt=create_example_prompt(),
        prefix=read_text("treatment_checker/query_routing/prefix.txt"),
        suffix=read_text("treatment_checker/query_routing/suffix.txt"),
        input_variables=["question"],
        example_separator="\n",
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    return prompt


def generate_examples() -> list[dict]:
    """
    Several simple examples of desired routing responses.

    Examples are stored here as they are too short to justify their own files.
    """
    ex1 = {
        "question": "I need drug information for ibuprofen",
        "answer": "drug_db",
    }
    ex2 = {
        "question": "I need patient information for 12445",
        "answer": "patient_history_db",
    }
    ex3 = {
        "question": "Is aspirin suitable for patient 12234?",
        "answer": "patient_history_db",
    }
    ex4 = {
        "question": "I will prescribe Amoxicillin for patient 23231",
        "answer": "patient_history_db",
    }

    return [ex1, ex2, ex3, ex4]


def create_example_prompt() -> PromptTemplate:
    """
    Example question-answer pairs for few shot prompt for routing responses.
    """
    examples_template = "[Question]: {question}\n [Answer]: {answer}"
    examples_input = ["question", "answer"]
    return PromptTemplate(template=examples_template, input_variables=examples_input)
