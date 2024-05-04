from sqlalchemy.orm import Session
from domains.dataset import Dataset
from domains.model import Model
from json import dumps

def patch_analysis(request):

  return


def get_inactive_dataset(db: Session) -> str:
  datasets = db.query(Dataset).filter(Dataset.status == 'INACTIVE').all()
  dataset_list = []
  for dataset in datasets:
      dataset_dict = {
          "data": dataset.data,
          "job": dataset.job
      }
      dataset_list.append(dataset_dict)
  return dataset_list

def get_active_model(db: Session):
  model = db.query(Model).filter(Model.status == 'ACTIVE').one()
  model_path = model.file_path

  return model_path