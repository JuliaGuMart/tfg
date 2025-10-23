from langchain_groq import ChatGroq
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os

# Añadir la API key de Groq
os.environ["GROQ_API_KEY"] = "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"

# Crear la app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Crear el modelo
llm = ChatGroq(model="deepseek-r1-distill-llama-70b", temperature=0.7)

# Historial de conversación
conversation_history = [("system", "Eres un asistente util que ayuda a reservar billetes de tren. haz las preguntas necesarias de una en una y en un orden lógico")]

# Función para obtener respuesta
def get_response(user_input):
    conversation_history.append(("human", user_input))
    ai_message = llm.invoke(conversation_history)
    print(ai_message)
    conversation_history.append(("ai", ai_message.content))
    return str(ai_message.content.split('</think>\n\n')[-1])  # Convertir el mensaje a cadena de texto

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_input = data.get('message')
    response = get_response(user_input)
    return JSONResponse(content={'response': response})

@app.get("/")
async def index():
    return FileResponse('static/index.html')

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8080)
