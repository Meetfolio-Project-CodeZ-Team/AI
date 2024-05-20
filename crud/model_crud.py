from sqlalchemy.orm import Session
from domains.model import Model
from datetime import datetime

def get_model(db: Session, model_id: int):
  return db.query(Model).filter(Model.model_id == model_id).one()


def change_model_status(db: Session, to_model_id: int, from_model_id: int):
    model_to_active = get_model(db, to_model_id)
    model_to_inactive = get_model(db, from_model_id)
    if model_to_active and model_to_inactive:
        model_to_active.status = 'ACTIVE'
        model_to_inactive.status = 'INACTIVE'
        db.commit()
        return model_to_active
    else:
        return "Model Not Found"

def soft_delete_model(db: Session, model_id: int):
  model_to_delete = get_model(db, model_id)
  if model_to_delete and model_to_delete.version_status == 'DEPRECATED':
    model_to_delete.version_status = 'OBSOLETE'
    db.commit()
  else:
      return "Model Not Found"


