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

settings = Settings()

# db = {
#     'user'     : 'root',
#     'password' : 'meetfolio123',
#     'host'     : '34.22.92.164',
#     'port'     : '3306',
#     'database' : 'meetfolio_dev'
# }

# DB_URL = f"mysql+pymysql://{db['user']}:{db['password']}@{db['host']}:{db['port']}/{db['database']}"