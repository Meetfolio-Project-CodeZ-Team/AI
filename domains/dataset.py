from sqlalchemy import Table, Column, Bignteger, String, DateTime, Text, Enum
from db.session import Base, engine, metadata

dataset_table = Table('dataset', metadata, autoload_with=engine)

class Dataset(Base):
    __tablename__ = 'dataset'

    dataset_id = Column(BigInteger, primary_key=True, autoincrement=True)
    data = Column(Text, nullable=False)
    domain = Column(String(255), nullable=True)
    url = Column(String(255), nullable=True)
    job = Column(Enum('백엔드', '웹개발', '앱개발', '디자인', 'AI'), nullable=False)
    status = Column(Enum('ACTIVE', 'INACTIVE'), default='ACTIVE', nullable=False)
    created_at = Column(DateTime(6), nullable=False)
    updated_at = Column(DateTime(6), nullable=False)
