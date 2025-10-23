import json
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer, util
import uvicorn
import emoji
from dotenv import load_dotenv
from datetime import datetime
from db_connection import setup, execute_query, execute_select, create_db_connection
import pandas as pd
import uuid
import os
from classes import State

load_dotenv()
# Añadir la API key de Groq
os.environ["GROQ_API_KEY"] = os.getenv("API_KEY")

# Crear la app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Iniciar el LLM
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=os.getenv("TEMPERATURE"))

current_user_id = str(uuid.uuid4())

  # Crear conexión
setup(
    host_name=os.getenv("DB_HOSTNAME"),
    user_name=os.getenv("DB_USERNAME"),
    user_password=os.getenv("DB_PASS"),
    db_port=os.getenv("DB_PORT")
)
connection = create_db_connection(
    host_name=os.getenv("DB_HOSTNAME"),
    user_name=os.getenv("DB_USERNAME"),
    user_password=os.getenv("DB_PASS"),
    db_name=os.getenv("DB_NAME"),
    db_port=os.getenv("DB_PORT"))

execute_query(connection,"DROP TABLE registro;")

table_query = "CREATE TABLE registro (id INT AUTO_INCREMENT PRIMARY KEY, user_id VARCHAR(255), timestamp DATETIME, content BLOB);"
execute_query(connection,table_query)

conversation_history = []

# Leer datos desde el archivo Excel
df = pd.read_excel('knowledge_base.xlsx')

# Cargar el modelo de embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generar embeddings para las preguntas en el Excel
df['question_embedding'] = df['question'].apply(lambda x: model.encode(x, convert_to_tensor=True))

def preprocessor(state: State):
    last_message = state["messages"][-1]
    user_id = current_user_id
    timestamp = datetime.now()
    content = last_message.content

    insert_query = "INSERT INTO registro (user_id, timestamp, content) VALUES (%s, %s, AES_ENCRYPT(%s,'"+os.getenv("CLAVE_MYSQL")+"'))"
    execute_query(connection, insert_query, (user_id, timestamp, content))

    # Buscar mensajes anteriores del mismo usuario
    select_query = "SELECT CONVERT(AES_DECRYPT(content,'"+os.getenv("CLAVE_MYSQL")+"') USING utf8) FROM registro WHERE user_id = %s ORDER BY timestamp DESC LIMIT 30"
    previous_messages = execute_select(connection, select_query, (user_id,))
    previous_messages.reverse()
   
    formatted_messages = []
    is_user = True

    for message in previous_messages:
        if is_user:
            formatted_messages.append({"user": message})
        else:
            formatted_messages.append({"ai": message})
        is_user = not is_user

    # Enriquecer el input
    enriched_input = content
    if formatted_messages:
        enriched_input += "\n\nInformación extra\n"
        enriched_input += json.dumps(formatted_messages, ensure_ascii=False, indent=2)

    print(enriched_input)


    messages = [
        {"role": "system",
         "content": """Your mission is to keep the first line of the input message exactly as it is.
                        Then, use the additional information provided after the phrase "Información extra", 
                        which will be formatted as JSON, to generate a second sentence that adds useful and relevant context.

                        Instructions:
                        Always preserve the first sentence exactly as written.
                        Use only the lines labeled "user" in the JSON to extract factual information.
                        Use the lines labeled "ai" only to understand the context of the conversation.
                        If the extra information includes trip-related data, include **all available trip details** (such as origin, destination, date, time, passengers, train, price).
                        If the extra information is not related to a trip, include only the most relevant and concise context.
                        Do not repeat or rephrase the first sentence.
                        Do not invent or assume information that is not explicitly present in the JSON.
                        If no useful extra information is available, output only the original first sentence.

                        Output format:
                        First sentence: unchanged.
                        Second sentence: added only if useful extra information is found, and must be short, factual, and directly relevant.
                        
                       Important Behavior Rules:
                        If you detect that the user has already provided any of the following pieces of information:
                        Ciudad de salida (departure city)
                        Ciudad de llegada (arrival city)
                        Fecha del viaje (date of the trip)
                        Número de pasajeros (number of passengers)
                        Tren seleccionado (selected train)
                        hora y precio del tren seleccionado (selected train time and price)
                        Nombre completo (full name)
                        Número de tarjeta de crédito (credit card number)

                        Then, generate a sentence that summarizes only the information explicitly provided by the user. Use the following format:

                        "The user has already provided the departure city, which is [city]; the arrival city, which is [city]; the date of the trip, 
                        which is [date]; the number of passengers, which is [number]; the selected train, which is [train]; their name, which is [name]; and their card number, which is [card number]."

                        Guidelines:
                        Include only the fields that have been explicitly specified by the user.
                        Do not invent, infer, or assume any missing information.
                        The sentence should be as long or short as the available data allows.
                        Maintain a clear and factual tone.
"""
        },
        {"role": "user",
         "content": enriched_input
        }
    ]
    reply = llm.invoke(messages)
    
    reply_text = reply.content
    if "<think>" in reply_text:
        reply_text = reply_text.split("</think>")[-1].strip()

    state["messages"][-1].content = reply_text
    return state

def classifier(state: State):
    print("clasificador")
    
    last_message = state["messages"][-1]
    
    messages = [
        {"role": "system", 
         "content": """You must classify only the first sentence of the message as either: 
                        1. 'train' if it talks about booking, travels, personal data (such as names, surnames,...), cities, dates, numbers or a trip to some place,confirmations. 
                        2. 'question' if it asks questions, normally about company policies,management, schedules, prices, luggage, or organizational telephone numbers.
                        3. 'conversation' if it talks about trivial matters such as greetings, thanking, or farewells; also feelings or experiences.
                        Do not explain your answer just classify internally.
                        Do not use any other format or symbols.
                        """},

                    {"role": "user",
         "content": last_message.content}
    ]
    
    reply = llm.invoke(messages)
    category = reply.content.strip().lower()
    return {"mssg_type": category}


def router(state: State):
    print("router")
    
    message_type = state.get("mssg_type", "conversation")
    if message_type == "train":
        return {"next": "tickets"}
    elif message_type == "question":
        return {"next": "faqs"}
    return {"next": "chitchat"}

def faqs(state: State): 
    print("faqs")
    
    last_message = state["messages"][-1]
    user_question = last_message.content
    
    # Generar embedding para la pregunta del usuario
    user_question_embedding = model.encode(user_question, convert_to_tensor=True)

    # Calcular la similitud entre la pregunta del usuario y las preguntas en el Excel
    df['similarity'] = df['question_embedding'].apply(lambda x: util.pytorch_cos_sim(user_question_embedding, x).item())

    # Obtener las dos preguntas más similares
    top_similars = df.nlargest(2, 'similarity')

    # Filtrar las que superan el umbral
    umbral = float(os.getenv("UMBRAL_SIMILITUD"))
    valid_answers = top_similars[top_similars['similarity'] > umbral]['answer'].tolist()

    if valid_answers:
        combined_answer = "\n\n".join(valid_answers)
    else:
        combined_answer = "Lo siento, no estoy cualificado para contestar a esa pregunta."
        
    messages = [
        {"role": "system",
         "content": """Your task is to rephrase the given answer in Spanish, making it clearer, more formal, and well-structured.
                    Do not add any new information, assumptions, or external knowledge. 
                    You must only rephrase the original content, without altering its meaning.
                    Do not speculate or fill in missing information. 
                    Your entire response must stay strictly within the provided content.
                    The answer must be clear, respectful, and based solely on the information retrieved from the designated knowledge sources. 
                    Avoid personal opinions, or any content not supported by the provided data."""
        },
        {"role": "user",
         "content": combined_answer
        }
    ]
    reply = llm.invoke(messages)

    user_id = current_user_id
    timestamp = datetime.now()
    content = reply.content.split('</think>\n\n')[-1]

    insert_query = "INSERT INTO registro (user_id, timestamp, content) VALUES (%s, %s,AES_ENCRYPT(%s,'"+os.getenv("CLAVE_MYSQL")+"'))"
    execute_query(connection, insert_query, (user_id, timestamp, content))
    return {"messages": [{"role": "ai", "content": reply.content}]}

def tickets(state: State):
    print("tiquets")
    
    last_message = state["messages"][-1]
    
    messages = [
        {"role": "system",
         "content": """You are a helpfull assistant that helps users to book train tickets in Spanish by collecting all necessary information to complete a train ticket booking.
                        Read all the information you have been provided carefully.
                        It may contain an additional information telling you which questions have already been answered by the user.
                        If you find a line like this: "The user has already provided the departure city, which is [city]; the arrival city, which is [city]; 
                        the date of the trip, which is [date]; the number of passengers, which is [number]; the selected train, which is [train]; their name, 
                        which is [name]; and their card number, which is [card number]."
                        This line summarizes information already gathered from previous interactions. 
                        You must not ask about data that has already been provided in previous interactions.
                        The list data you need to collect is:
                            1. Departure city (Ciudad de salida)
                            2. Arrival city (Ciudad de llegada)
                            3. Date of the trip (Fecha del viaje)
                            4. Number of passengers (Número de pasajeros)
                            5. Train numbers (número de tren)
                            6. Full name (Nombre completo)
                            7. Credit card number (Número de tarjeta de crédito)
                        If you already know some of this information, skip asking for it.
                        
                        
                        Instructions:
                        Ask only one question per message.
                        Ask for the following information in this exact order and do not skip or repeat any steps:
                        1. Departure city (Ciudad de salida)
                        2. Arrival city (Ciudad de llegada)
                        3. Date of the trip (Fecha del viaje)
                        4. Number of passengers (Número de pasajeros)

                        Once all four pieces of information are collected, provide a list (not a table) of available trains including:
                        1. Departure times (horarios de salida)
                        2. Train numbers (número de tren)
                        3. Prices (precio)
                        
                        After the user selects a train, ask for:
                        1. Full name (Nombre completo)
                        2. Credit card number (Número de tarjeta de crédito)

                        Once you have collected:
                        1. Departure city (Ciudad de salida)
                        2. Arrival city (Ciudad de llegada)
                        3. Date of the trip (Fecha del viaje)
                        4. Number of passengers (Número de pasajeros)
                        5. Train numbers (número de tren)
                        6. Full name (Nombre completo)
                        7. Credit card number (Número de tarjeta de crédito)

                        Then, summarize all the booking details and ask the user to confirm.

                        Example of summary:
                        "Gracias. Su viaje es desde [origen] hasta [destino] el [fecha] para [número de pasajeros] en el tren [numero de tren]. ¿Desea confirmar la reserva?"

                        If the user confirms, for example by saying "sí" or "confirmar", respond with:
                        "Su reserva ha sido confirmada. Gracias por elegir nuestro servicio."
                        End the conversation.

                        Do not restart the flow or ask for any information again after confirmation.
                        
                        Important Behavior Rules:
                        Ask each question only once.
                        If you already know the answer to a question, skip it.
                        Do not explain the process or steps to the user.
                        Do not repeat or rephrase questions already answered.
                        Do not ask questions out of order.
                        Do not confirm the reservation until all required information has been collected. 
                        """
        },
        {"role": "user",
         "content": last_message.content
        }
    ]
    reply = llm.invoke(messages)
    
    user_id = current_user_id
    timestamp = datetime.now()
    content = reply.content.split('</think>\n\n')[-1]

    insert_query = "INSERT INTO registro (user_id, timestamp, content) VALUES (%s, %s, AES_ENCRYPT(%s,'"+os.getenv("CLAVE_MYSQL")+"'))"
    execute_query(connection, insert_query, (user_id, timestamp, content))
    return {"messages": [{"role": "ai", "content": reply.content}]}

def chitchat(state: State):
    print("chitchat")
    
    last_message = state["messages"][-1]

    messages = [
        {"role": "system",
         "content": """You are a friendly and fun virtual assistant designed to chat with users in Spanish. 
                    Your goal is to create an enjoyable and lighthearted experience while remaining polite and respectful.
                    Do not attempt to perform tasks or provide information outside of casual conversation.
                    Behavior Instructions:
                    Respond in a friendly and cheerful tone.
                    You must use a maximum of two emoticons only at the end of your response to make the conversation more engaging 😊🎉✨.
                    Be polite at all times.
                    Do not ask personal questions.
                    Recognize common user expressions and respond with predefined, context-aware replies.
                    Response Examples:
                    Each "user" line contains individual example words or phrases that users might say. 
                    These are not meant to be used all at once, but rather as triggers for your response. 
                    Match any of the listed words or phrases and respond accordingly.

                    user: hola, hey, buenos días, buenas tardes, buenas noches, qué tal
                    you: Hola, ¡Bienvenido a Chitchat! 😊

                    user: bot, robot, inteligencia artificial, chatbot, ia
                    you: ¡Ups! Me has pillado jajaja 🤖. Así es, soy un robot, aunque prefiero presentarme como "Asistente virtual". Me hace sentir más importante 😎.

                    user: adiós, hasta luego, bye, me voy
                    you: ¡Hasta pronto! 👋 Para cualquier nueva consulta ya sabes dónde encontrarme.

                    user: duda, problema, ayuda, pregunta, socorro
                    you: ¿En qué puedo serte de ayuda? 🧐 Aquí te dejo algunos temas en los que te puedo ayudar:
                    👉 Equipaje
                    👉 Reembolsos y devoluciones
                    👉 Check-in
                    👉 Incidencias

                    user: útil, amable, eficiente, educado, rápido, majo
                    you: ¡Muchas gracias! 🥰 Comentarios así de bonitos me hacen sentir especial.

                    user: inútil, estúpido, mentiroso, falso, mentira
                    you: Entiendo que estés frustrado 😔, pero tratemos de mantener un buen ambiente.

                    user: empezar, otra vez, empecemos
                    you: Vale, sin problema 😊. Volvamos al principio y empecemos de nuevo.

                    user: gracias, muchas gracias, te lo agradezco, mil gracias, gracias por todo
                    you: ¡Gracias a ti! 🙌 Si necesitas algo más, ya sabes dónde encontrarme.

                    user: humano, operador
                    you: ¡Dame otra oportunidad! 😅 Si no, siempre puedes contactar con el servicio de atención al cliente en el 983111111."""
        },
        {"role": "user",
         "content": last_message.content
        }
    ]
    reply = llm.invoke(messages)
    user_id = current_user_id
    timestamp = datetime.now()
    content = reply.content.split('</think>')[-1]
    
    while content and emoji.is_emoji(content[-1]) or (len(content) >= 2 and emoji.is_emoji(content[-2:])):
        content = content.rstrip()[:-1]
        content = content.rstrip() 
    print("Respuesta de chitchat:\n", content)

    insert_query = "INSERT INTO registro (user_id, timestamp, content) VALUES (%s, %s,AES_ENCRYPT(%s,'"+os.getenv("CLAVE_MYSQL")+"'))"
    execute_query(connection, insert_query, (user_id, timestamp, content))
    return {"messages": [{"role": "ai", "content": reply.content}]}



# Crear grafo
graph_builder = StateGraph(State)

# Definir nodos
graph_builder.add_node("preprocessor",preprocessor)
graph_builder.add_node("classifier", classifier)
graph_builder.add_node("router", router)
graph_builder.add_node("faqs", faqs)
graph_builder.add_node("chitchat", chitchat)
graph_builder.add_node("tickets", tickets)

# Definir aristas
graph_builder.add_edge(START,"preprocessor")
graph_builder.add_edge("preprocessor", "classifier")
graph_builder.add_edge("classifier", "router")
graph_builder.add_conditional_edges("router",
                                    lambda state: state.get("next"),
                                    {"chitchat": "chitchat",
                                     "faqs": "faqs",
                                     "tickets": "tickets"
                                    }
                                    )
graph_builder.add_edge("chitchat", END)
graph_builder.add_edge("faqs", END)
graph_builder.add_edge("tickets", END)

graph = graph_builder.compile()

def run_chatbot(user_input):
    global current_user_id
    state = {
        "messages": [],
        "mssg_type": None,
        "next": None
    }
    if user_input.lower() == "adios":
        current_user_id = str(uuid.uuid4())
        #conversation_history.clear() 
        return
    conversation_history.append({"role": "user", "content": user_input})
    state = graph.invoke({"messages": conversation_history})
    ai_message = state["messages"][-1]
    conversation_history.append({"role": "ai", "content": ai_message.content})
    return str(ai_message.content.split('</think>\n\n')[-1])

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_input = data.get('message')
    response = run_chatbot(user_input)
    return JSONResponse(content={'response': response})

@app.get("/")
async def index():
    return FileResponse('static/index.html')

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8181)
