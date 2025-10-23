from typing import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from typing_extensions import TypedDict
import os

# Añadir la API key de Groq
os.environ["GROQ_API_KEY"] = "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"

# Iniciar el LLM
llm = ChatGroq(model="deepseek-r1-distill-llama-70b", temperature=0.7)

# Historial de conversación como lista de diccionarios
conversation_history = [{"role": "system", "content": "eres un asistente útil"}]

#aqui defino todos los atributos que quiero que tenga mi nodo
class State(TypedDict):
    current_node:str
    status:str
    messages: Annotated[list, add_messages]

# Definir un graph builder
graph_builder = StateGraph(State)

# Este nodo genera una respuesta llamando al llm, modifica el atributo de mensajes y el de nodo actual
def chatbot(state: State):
    response = llm.invoke(state["messages"])
    state["messages"].append({"role": "assistant", "content": response.content})
    state["current_node"] = "chatbot"
    return state


# Este nodo accede al ultimo mensaje recibido y si es alguna de las palabras clave actualiza el estado del nodo
def statusChange(state: State):
    last_message = state["messages"][-1]
    content = last_message["content"] if isinstance(last_message, dict) else last_message.content
    if "error" in content.lower():
        state["status"] = "ERROR"
        state["messages"].append({"role": "user", "content": "hay errores"})
        state["status"] = "RUNNING"
        response = llm.invoke(state["messages"])
        state["messages"].append({"role": "assistant", "content": response.content})
    elif "success" in content.lower():
        state["status"] = "SUCCESS"
        state["messages"].append({"role": "user", "content": "has completado exitosamente la tarea"})
        state["status"] = "RUNNING"
        response = llm.invoke(state["messages"])
        state["messages"].append({"role": "assistant", "content": response.content})
    else:
        state["status"] = "RUNNING"
    state["current_node"] = "status"
    return state


# Aqui creamos el grafo
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("status",statusChange)
graph_builder.add_edge(START,"chatbot")
graph_builder.add_edge("chatbot","status")
graph_builder.add_edge("status",END)

graph = graph_builder.compile()


while True:
    user_input = input("Enter a message: ")
    if user_input.lower() == "adios":
        break
    conversation_history.append({"role": "user", "content": user_input})
    state = graph.invoke({"messages": conversation_history})
    ai_message = state["messages"][-1]
    conversation_history.append({"role": "assistant", "content": ai_message["content"] if isinstance(ai_message, dict) else ai_message.content})
    print(f"Nodo actual: {state['current_node']}, Estado: {state['status']}")
    print(ai_message["content"] if isinstance(ai_message, dict) else ai_message.content)
