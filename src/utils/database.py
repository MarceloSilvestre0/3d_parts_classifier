from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Float, String, Integer, DateTime
from datetime import datetime


Base = declarative_base()

class chess(Base):

    __tablename__ = "chess_parts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    part_name = Column(String(30), nullable=False)
    dt_create = Column(DateTime)
    dt_modified = Column(DateTime)
    file_size = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.now)
    