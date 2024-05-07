from sqlalchemy.orm import Session
from domains.dataset import Dataset
from domains.model import Model
from domains.coverletter import Coverletter
from domains.analysis import Analysis
from json import dumps
from core.config import settings
from datetime import datetime
from sqlalchemy.orm.exc import NoResultFound

def patch_coverletter(db: Session, cover_letter_id, data):

  cover_letter = db.query(Coverletter).filter(Coverletter.cover_letter_id==cover_letter_id).first()
  if cover_letter:
    cover_letter.keyword_1 = data['keyword1']
    cover_letter.keyword_2 = data['keyword2']
    cover_letter.job_keyword = data['job_keyword']
    db.commit()
  else:
    return "Coverletter Not Found"

def save_analysis(db: Session, cover_letter_id, job_suitability, skill_keywords):
  analysis = Analysis(
    cover_letter_id = cover_letter_id,
    job_suitability=round(job_suitability,4),
    keyword_1=skill_keywords[0],
    keyword_2=skill_keywords[1],
    keyword_3=skill_keywords[2],
    created_at=datetime.now(),
    updated_at=datetime.now()
  )
  db.add(analysis)
  db.commit()

  return 1

def get_inactive_dataset(db: Session) -> str:
  datasets = db.query(Dataset).filter(Dataset.status == 'INACTIVE').all()
  dataset_list = []
  for dataset in datasets:
      dataset_dict = {
          "data": dataset.data,
          "job": dataset.job
      }
      dataset_list.append(dataset_dict)
  
  # 가져온 데이터셋 -> 'ACTIVE'로 벼경
  # for dataset in datasets:
  #   dataset.status = 'ACTIVE'
  return datasets, dataset_list

def get_active_model(db: Session):
  try:
    model = db.query(Model).filter(Model.status == 'ACTIVE').one()
    model_path = model.file_path

  except NoResultFound:
    model_path = '/home/t24105/v0.9src/ai/model/meetfolio_model_v1.pt'

  return model_path

def save_model(db: Session, file_name, version, accuracy, loss):

  model = Model(
    name = "meetfolio_model",
    file_name = file_name,
    file_path = settings.MODEL_PATH + file_name,
    version = version,
    accuracy = accuracy,
    loss = loss,
    created_at=datetime.now(),
    updated_at=datetime.now()
  )
  db.add(model)
  db.commit()

  return model.model_id