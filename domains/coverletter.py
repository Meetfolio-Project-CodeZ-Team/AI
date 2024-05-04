from sqlalchemy import Table, Column, BigInteger, Text, String, Enum
from db.session import Base, engine, metadata

coverletter_table = Table('cover_letter', metadata, autoload_with=engine)

class Coverletter(Base):
    __tablename__ = coverletter_table

    cover_letter_id = Column(BigInteger, primary_key=True)
    answer = Column(Text, nullable=False)
    keyword_1 = Column(String(255), nullable=True)
    keyword_2 = Column(String(255), nullable=True)
    question = Column(String(255), nullable=False)
    job_keyword = Column(Enum('BACKEND', 'WEB', 'APP', 'DESIGN', 'AI'), nullable=False)