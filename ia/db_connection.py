import mysql.connector 
from mysql.connector import Error


def setup(host_name, user_name, user_password, db_port):
    connection = None
    try:
        connection = mysql.connector.connect(
            host=host_name,
            user=user_name,
            passwd=user_password,
            port=db_port
        )
        print("MySQL Database connection successful")
        query = "CREATE DATABASE historial;"
        execute_query(connection,query)

    except Error as err:
        print(f"Error al crear la conexión a la base de datos: '{err}'")

def create_db_connection(host_name, user_name, user_password, db_name,db_port):
    connection = None
    try:
        connection = mysql.connector.connect(
            host=host_name,
            user=user_name,
            passwd=user_password,
            database = db_name,
            port=db_port
        )
        print("MySQL Database connection successful")
    except Error as err:
        print(f"Error al conectar a la base de datos: '{err}'")

    return connection


def execute_query(connection, query, params=None):
    cursor = connection.cursor()
    try:
        cursor.execute(query, params)
        connection.commit()
        print("Query successful")
    except Error as err:
        print(f"Error: '{err}'")




def execute_select(connection, query, params=None):
    cursor = connection.cursor()
    try:
        cursor.execute(query, params)
        return cursor.fetchall()
    except Error as err:
        print(f"Error: '{err}'")
        return []

