from typing import Annotated, Literal
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

# Definir la estructura de cada estado del grafo
class State(TypedDict):
    messages: Annotated[list, add_messages]
    mssg_type: str | None
    next: str | None
