from sqlalchemy import Table, Column, BigInteger, Integer, String, DateTime, Enum, Float
from db.session import Base, engine, metadata

analysis = Table('analysis', metadata, autoload_with=engine)

class Analysis(Base):
  __tablename__ = 'analysis'

  analysis_id = Column(BigInteger, primary_key=True, autoincrement=True)
  job_suitability = Column(Float, nullable=False)
  satisfaction = Column(Integer, nullable=True)
  cover_letter_id = Column(BigInteger, nullable=False)
  keyword_1 = Column(String(255), nullable=False)
  keyword_2 = Column(String(255), nullable=False)
  keyword_3 = Column(String(255), nullable=False)
  status = Column(Enum('ACTIVE', 'INACTIVE'), default='ACTIVE', nullable=False)
  created_at = Column(DateTime(6), nullable=False)
  updated_at = Column(DateTime(6), nullable=False)