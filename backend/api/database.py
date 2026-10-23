
import os

import mysql.connector
from dotenv import load_dotenv

# Load the .env file from the project root.
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))


def get_db_connection():
    """Open and return a MySQL database connection."""
    return mysql.connector.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "3306")),
        database=os.getenv("DB_NAME", "medical_image_assistant"),
        user=os.getenv("DB_USER", "medical_app"),
        password=os.getenv("DB_PASSWORD", ""),
        connection_timeout=5,
    )
