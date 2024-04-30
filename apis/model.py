from sqlalchemy.orm import Session
from domains.model import Model

def test_index(db: Session):

  return db.query(Model).all()