from typing import Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
import os

# Añadir la API key de Groq
os.environ["GROQ_API_KEY"] = "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"

# Iniciar el LLM
llm = ChatGroq(model="deepseek-r1-distill-llama-70b", temperature=0.7)

# Historial de conversación como lista de diccionarios
conversation_history = [{"role": "system", "content": "eres un asistente útil"}]

# Crear el estado (tipo de información que queremos manejar en el grafo)
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Definir un graph builder
graph_builder = StateGraph(State)

# Crear los nodos del grafo
def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


# Crear aristas entre los nodos en el formato Origen-Destino
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()

while True:
    user_input = input("Enter a message: ")
    if (user_input == "Adios"):
        break
    conversation_history.append({"role": "user", "content": user_input})
    state = graph.invoke({"messages": conversation_history})
    ai_message = state["messages"][-1]
    conversation_history.append({"role": "ai", "content": ai_message.content})
    print(ai_message.content)
