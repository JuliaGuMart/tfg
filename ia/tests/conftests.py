import os
from dotenv import load_dotenv

# Carga el .env del directorio 'ia'
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))