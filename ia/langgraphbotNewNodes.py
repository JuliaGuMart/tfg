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

# El tipo del mensaje va a ser un valor literal (~enum) y eso equivale a un field de pydantic que representa lo que hay en la descripción
class MessageClassifier(BaseModel):
    message_type: Literal["emocional","logical"] = Field(
        ...,
        description="Classify if the message requires an emocional (therapist) or logical response"
    )

# Ahora esta clase no solo tiene mensajes sino que ademas guarda el tipo del mensaje
class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_types:str | None
    next:str|None

def boss(state:State):
    last_message = state["messages"][-1]
    messages = [
        {"role":"system",
         "content":"Your mission is to take the message and divide it in parts based on the topic they are about and reorganize it putting together the sentences with the same topic"
        },
        {"role":"user",
         "content": last_message.content
        }
    ]
    reply = llm.invoke(messages)
    return {"messages": [{"role":"ai","content":reply.content}]}

def classify_message(state:State):
    last_message = state["messages"][-1]
    # crea una version concreta del llm que genera una respuesta segun la estructura que le pasamos
    classifier_llm =llm.with_structured_output(MessageClassifier)
    result = classifier_llm.invoke([
        {"role":"system", 
         "content":"classify the message as either: 'emotional' if it ask for emotional support, feelings or personality and 'logical' if it asks for facts, information, analysis or practical solutions"
        },
        {"role":"user",
         "content":last_message.content
        }
    ])
    return {"message_type":result.message_type}

def therapist_agent(state:State):
    last_message = state["messages"][-1]
    messages = [
        {"role":"system",
         "content":"help the user to the best of your habilities, be kind and considerate"
        },
        {"role":"user",
         "content": last_message.content
        }
    ]
    reply = llm.invoke(messages)
    return {"messages": [{"role":"ai","content":reply.content}]}

def logical_agent(state:State):
    last_message = state["messages"][-1]
    messages = [
        {"role":"system",
         "content":"help the user to the best of your habilities, be serius and objective"
        },
        {"role":"user",
         "content": last_message.content
        }
    ]
    reply = llm.invoke(messages)
    return {"messages": [{"role":"ai","content":reply.content}]}

def router(state:State):
    message_type = state.get("message_type","logical")
    if message_type == "emocional":
        return {"next":"therapist"}
    return{"next":"logical"}

graph_builder = StateGraph(State)

graph_builder.add_node("classifier",classify_message)
graph_builder.add_node("router",router)
graph_builder.add_node("therapist",therapist_agent)
graph_builder.add_node("logical",logical_agent)

graph_builder.add_edge(START,"classifier")
graph_builder.add_edge("classifier","router")
graph_builder.add_conditional_edges("router",
                                    lambda state:state.get("next"),
                                    {"therapist":"therapist",
                                     "logical":"logical"
                                    }
                                    )
graph_builder.add_edge("therapist",END)
graph_builder.add_edge("logical",END)

graph = graph_builder.compile()

def run_chatbot():
    state = {"messages":[],
             "message_type":None,
             "next": None
            }
    while True:
        user_input = input("Message: ")
        if user_input.lower() == "adios":
            break
        # miramos el estado actual y vemos los mensajes y añadimos el nuevo mensaje
        state["messages"]=state.get("messages",[])+[{"role":"user", "content":user_input}]
        state = graph.invoke(state)
        if state.get("messages") and len(state["messages"])>0:
            last_message =state["messages"][-1]
            print({last_message.content})


