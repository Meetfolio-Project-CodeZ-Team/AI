import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
  DB_USERNAME = os.getenv("MYSQL_USER")
  DB_PASSWORD = os.getenv("MYSQL_PASSWORD")
  DB_HOST = os.getenv("MYSQL_HOST")
  DB_PORT = os.getenv("MYSQL_PORT")
  DB_DATABASE = os.getenv("MYSQL_DB")
  DB_URL = f'mysql+pymysql://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_DATABASE}'

  GPT_KEY=os.getenv("GPT_KEY")

  CLOVA_ID = os.getenv("CLOVA_URL")
  CLOVA_SECRET = os.getenv("CLOVA_SECRET")
  CLOVA_URL = os.getenv("CLOVA_URL")

  KOBERT_DEFAULT=os.getenv("KOBERT_DEFAULT")

  MODEL_PATH=os.getenv("MODEL_PATH")
  MODEL_NAME=os.getenv("MODEL_NAME")

settings = Settings()