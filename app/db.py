import os
import mysql.connector
from pymongo import MongoClient
from urllib.parse import quote_plus

def mysql_conn():
    try:
        conn = mysql.connector.connect(
            host=os.getenv("MYSQL_HOST", "mysql"),
            port=int(os.getenv("MYSQL_PORT", 3306)),
            user=os.getenv("MYSQL_USER", "chat"),
            password=os.getenv("MYSQL_PASSWORD", "chatpass"),
            database=os.getenv("MYSQL_DB", "shopping"),
            autocommit=True
        )
        return conn
    except mysql.connector.Error as err:
        print(f"Error connecting to MySQL: {err}")
        raise

def mongo_conn():
    uri = os.getenv("MONGO_URI")
    if not uri:
        print("Error: MONGO_URI environment variable not set.")
        raise ValueError("MONGO_URI not configured")

    try:
        client = MongoClient(uri)
        client.admin.command('ismaster')

        db_name = uri.rsplit('/', 1)[-1].split('?', 1)[0]
        if not db_name or db_name.lower() == 'admin':
            db_name = os.getenv("MONGO_DEFAULT_DB", "university")
            print(f"Using default MongoDB database: {db_name}")

        return client[db_name]

    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        raise