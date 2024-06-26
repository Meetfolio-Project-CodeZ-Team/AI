from sqlalchemy import Table, Column, BigInteger, String, DateTime, Text, Enum, Float, Numeric
from db.session import Base, engine, metadata

model = Table('model', metadata, autoload_with=engine)

class Model(Base):
  __tablename__ = 'model'

  model_id = Column(BigInteger, primary_key=True, autoincrement=True)
  name = Column(String(255), nullable=False)
  file_name = Column(String(255), nullable=False)
  file_path = Column(String(255), nullable=False)
  accuracy = Column(Float, nullable=False)
  loss = Column(Float, nullable=False)
  version = Column(String(255), nullable=False)
  status = Column(Enum('ACTIVE', 'INACTIVE'), default='INACTIVE', nullable=False)
  activated_date = Column(DateTime(6), nullable=False)
  version_status = Column(Enum('DEPRECATED', 'OBSOLETE'), nullable=True)
  created_at = Column(DateTime(6), nullable=False)
  updated_at = Column(DateTime(6), nullable=False)