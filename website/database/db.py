import os
from pathlib import Path
from dotenv import load_dotenv
import mysql.connector

# Always load the .env from the website directory regardless of CWD
ENV_PATH = Path(__file__).resolve().parents[1] / '.env'
load_dotenv(dotenv_path=ENV_PATH)

def get_db_connection():
    host = os.getenv('DB_HOST', 'localhost')
    user = os.getenv('DB_USER', 'root')
    password = os.getenv('DB_PASSWORD', 'prajwaltp')
    database = os.getenv('DB_NAME', 'cardio_care')
    port = int(os.getenv('DB_PORT', '3306'))
    return mysql.connector.connect(host=host, user=user, password=password, database=database, port=port)


