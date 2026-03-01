from typing import TypedDict, Annotated
from langgraph.graph import add_messages

class AgentState(TypedDict):
    brand_description: str
    trend: str
    trend_description: str
    draft: str
    rewrites: int
    critic_review: dict
    approval : bool
    messages: Annotated[list, add_messages]