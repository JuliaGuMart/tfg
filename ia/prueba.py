import json
import os

from dotenv import load_dotenv
from db_connection import create_db_connection, execute_select, setup

load_dotenv()
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

select_query = "SELECT content FROM registro ORDER BY timestamp DESC LIMIT 15"
previous_messages = execute_select(connection, select_query)
json_text= ""
is_user=True

for message in previous_messages:
    if(is_user):
        formatted_message={"user":message}
        json_text+=json.dumps(formatted_message)+","
        is_user= not is_user
    else:
        formatted_message={"ai":message}
        json_text+=json.dumps(formatted_message)+","
        is_user= not is_user

print(json_text)