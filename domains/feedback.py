from sqlalchemy import Table, Column, Integer, String, DateTime, Text, Enum
from db.session import Base, engine, metadata

feedback_table = Table('feedback', metadata, autoload_with=engine)

class Feedback(Base):
    __tablename__ = feedback_table

    solution_id = Column(Integer, primary_key=True)
    cover_letter_id = Column(Integer, nullable=False, unique=True)
    solution_id = Column(Integer, primary_key=True, autoincrement=True)
    correction = Column(Text, nullable=False)
    recommend_question_1 = Column(String(255), nullable=False)
    recommend_question_2 = Column(String(255), nullable=False)
    recommend_question_3 = Column(String(255), nullable=False)
    status = Column(Enum('ACTIVE', 'INACTIVE'), default='ACTIVE', nullable=False)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)