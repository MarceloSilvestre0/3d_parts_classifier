import time
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os
from src.utils.database import Base
from src.core.bouding_box import bouding_box

load_dotenv()

POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_PORT = os.getenv("POSTGRES_PORT")
POSTGRES_DB = os.getenv("POSTGRES_DB")


DATABASE_URL = (
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)


engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)

def create_table():

    Base.metadata.create_all(engine)
    print("Tabela criada ou verificada com sucesso!")
    

#Salvando os dados no PostgreSQL
def save_data(data):
    """Salva os dados no banco PostgreSQL."""
    session = Session()
    novo_registro = bouding_box(data)
    session.add(novo_registro)
    session.commit()
    session.close()