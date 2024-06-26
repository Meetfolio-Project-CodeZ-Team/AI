from sqlalchemy import Table, Column, BigInteger, String, DateTime, Text, Enum
from db.session import Base, engine, metadata

dataset = Table('dataset', metadata, autoload_with=engine)

class Dataset(Base):
    __tablename__ = 'dataset'

    dataset_id = Column(BigInteger, primary_key=True, autoincrement=True)
    data = Column(Text, nullable=False)
    domain = Column(String(255), nullable=True)
    url = Column(String(255), nullable=True)
    job = Column(Enum('BACKEND', 'WEB', 'APP', 'DESIGN', 'AI'), nullable=False)
    status = Column(Enum('ACTIVE', 'INACTIVE'), default='INACTIVE', nullable=False)
    created_at = Column(DateTime(6), nullable=False)
    updated_at = Column(DateTime(6), nullable=False)
