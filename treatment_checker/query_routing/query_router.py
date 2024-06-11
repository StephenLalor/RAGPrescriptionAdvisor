"""
Contains QueryRouter dataclass defining the format the LLM response should adhere to.
"""

from typing import Literal

from langchain_core.pydantic_v1 import BaseModel, Field


class QueryRouter(BaseModel):
    """
    Data class specifying desired output format for query routing.
    """

    datasource: Literal["hist_chain", "diag_chain", "full_chain"] = Field(
        ...,
        description="Select most relevent chain for question.",
    )
