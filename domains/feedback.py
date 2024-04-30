from sqlalchemy import Table, Column, Integer
from db.session import Base, engine, metadata

feedback_table = Table('feedback', metadata, autoload_with=engine)

class Feedback(Base):
    __tablename__ = feedback_table

    solution_id = Column(Integer, primary_key=True)