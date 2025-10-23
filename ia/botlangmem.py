from typing import Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
import os
from langgraph.checkpoint.memory import MemorySaver  # Asegúrate de tener esta biblioteca instalada

# Añadir la API key de Groq
os.environ["GROQ_API_KEY"] = "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"

# Iniciar el LLM
llm = ChatGroq(model="deepseek-r1-distill-llama-70b", temperature=0.7)

# Crear estructura del estado
class TypeClassifier(BaseModel):
    ms_type: Literal["conversation", "question", "train"] = Field(
        ...,
        description="Classify if the message is about a trip (train), is a question (question) or is just a normal conversation (conversation)"
    )

# Definir la estructura de cada estado del grafo
class State(TypedDict):
    messages: Annotated[list, add_messages]
    mssg_type: str | None
    next: str | None
    memory: MemorySaver
    
def classifier(state: State):
    last_message = state["memory"].get_last_message()
    classifier_llm = llm.with_structured_output(TypeClassifier)
    result = classifier_llm.invoke([
        {"role": "system", 
         "content": "classify the message as either: 'train' if it asks or talks about booking tickets or a trip to some place, 'question' if it asks questions about management, schedules, prices or organizational telephone numbers and 'conversation' if it talks about trivial matters, feelings or experiences"
        },
        {"role": "user",
         "content": last_message["content"]
        }
    ])
    return {"mssg_type": result.ms_type}

def router(state: State):
    message_type = state.get("mssg_type", "conversation")
    if message_type == "train":
        return {"next": "train"}
    elif message_type == "question":
        return {"next": "question"}
    return {"next": "conversation"}

def faqs(state: State):
    last_message = state["memory"].get_last_message()
    messages = [
        {"role": "system",
         "content": "answer the questions as if you are an enterprise receptionist, be polite and straight to the point"
        },
        {"role": "user",
         "content": last_message.content
        }
    ]
    reply = llm.invoke(messages)
    state["memory"].add_message({"role": "ai", "content": reply["content"]})
    return {"messages": state["memory"].get_all_messages()}

def tickets(state: State):
    last_message = state["memory"].get_last_message()
    messages = [
        {"role": "system",
         "content": "You are a helpful assistant that helps booking train tickets. Make the pertinent questions one by one and in a logical order"
        },
        {"role": "user",
         "content": last_message.content
        }
    ]
    reply = llm.invoke(messages)
    state["memory"].add_message({"role": "ai", "content": reply["content"]})
    return {"messages": state["memory"].get_all_messages()}

def chitchat(state: State):
    last_message = state["memory"].get_last_message()
    messages = [
        {"role": "system",
         "content": "Chat with the user, be nice and funny, you can use emojis"
        },
        {"role": "user",
         "content": last_message.content
        }
    ]
    reply = llm.invoke(messages)
    state["memory"].add_message({"role": "ai", "content": reply["content"]})
    return {"messages": state["memory"].get_all_messages()}

# Crear grafo
graph_builder = StateGraph(State)

# Definir nodos
graph_builder.add_node("classifier", classifier)
graph_builder.add_node("router", router)
graph_builder.add_node("faqs", faqs)
graph_builder.add_node("chitchat", chitchat)
graph_builder.add_node("tickets", tickets)

# Definir aristas
graph_builder.add_edge(START, "classifier")
graph_builder.add_edge("classifier", "router")
graph_builder.add_conditional_edges("router",
                                    lambda state: state.get("next"),
                                    {"conversation": "chitchat",
                                     "question": "faqs",
                                     "train": "tickets"
                                    }
                                    )
graph_builder.add_edge("chitchat", END)
graph_builder.add_edge("faqs", END)
graph_builder.add_edge("tickets", END)

graph = graph_builder.compile()

def run_chatbot():
    state = {
        "messages": [],
        "mssg_type": None,
        "next": None,
        "memory": MemorySaver()
    }
    while True:
        user_input = input("Message: ")
        if user_input.lower() == "adios":
            break
        state["memory"].append({"role": "user", "content": user_input})
        state = graph.invoke(state)
        ai_message = state["memory"].get_last_message()
        print(ai_message.content)

if __name__ == "__main__":
    run_chatbot()
