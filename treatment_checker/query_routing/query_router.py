"""
Contains QueryRouter dataclass defining the format the LLM response should adhere to.
"""

from typing import Literal

from langchain_core.pydantic_v1 import BaseModel, Field


class QueryRouter(BaseModel):
    """
    Data class specifying desired output format for query routing.
    """

    datasource: Literal["drug_db", "patient_history_db"] = Field(
        ...,
        description="Select the most relevent datasource for the question.",
    )
