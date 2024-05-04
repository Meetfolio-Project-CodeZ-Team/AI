from sqlalchemy.orm import Session
from domains.dataset import Dataset
from json import dumps

def patch_analysis(request):


  return


def get_active_dataset(db: Session) -> str:
  datasets = db.query(Dataset).filter(Dataset.status == 'ACTIVE').all()
  dataset_list = []
  for dataset in datasets:
      dataset_dict = {
          "data": dataset.data,
          "job": dataset.job
      }
      dataset_list.append(dataset_dict)
  return dataset_list
