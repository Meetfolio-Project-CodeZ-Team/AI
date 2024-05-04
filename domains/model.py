from sqlalchemy import Table, Column, BigInteger, String, DateTime, Text, Enum, Float, Numeric
from db.session import Base, engine, metadata

model_table = Table('model', metadata, autoload_with=engine)

class Model(Base):
  __tablename__ = model_table

  model_id = Column(BigInteger, primary_key=True, autoincrement=True)
  name = Column(String(255), nullable=False)
  file_name = Column(String(255), nullable=False)
  file_path = Column(String(255), nullable=False)
  accuracy = Column(Float, nullable=False)
  loss = Column(Float, nullable=False)
  version = Column(Numeric(precision=38, scale=2), nullable=False)
  status = Column(Enum('ACTIVE', 'INACTIVE'), default='ACTIVE', nullable=False)
  learned_date = Column(DateTime(6), nullable=False)
  activated_date = Column(DateTime(6), nullable=False)
  created_at = Column(DateTime(6), nullable=False)
  updated_at = Column(DateTime(6), nullable=False)