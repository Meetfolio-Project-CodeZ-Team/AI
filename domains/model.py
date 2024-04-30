from sqlalchemy import Table
from db.session import Base, engine, metadata

model_table = Table('model', metadata, autoload_with=engine)

class Model(Base):
  __table__ = model_table