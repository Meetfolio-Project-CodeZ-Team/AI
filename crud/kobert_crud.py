from sqlalchemy.orm import Session
from domains.dataset import Dataset
from json import dumps

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

def parse_train_data_label(datasets):
   
   return
   
### 한글 전처리
# def konlpy_okt(df):
#     okt = Okt()
#     df['data'] = df['data'].map(lambda x : ' '.join(okt.morphs(x, stem=True)))
#     return df

#### 백엔드 -> 0 으로 변환 함수
def label_to_int(df):

  label_mapping = {"백엔드": 0, "웹개발": 1, "앱개발": 2, "AI": 3, "디자인": 4}
  df['label'] = df['label'].map(label_mapping)

### 훈련 & 테스트 데이터 분리
def data_train_test_split(df):
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

    train_texts = train_df['data'].astype(str).tolist()
    train_labels = train_df['label'].tolist()
    test_texts = test_df['data'].astype(str).tolist()
    test_labels = test_df['label'].tolist()

    train_labels_2d = [[label] for label in train_labels]
    test_labels_2d = [[label] for label in test_labels]

    encoder = OneHotEncoder(categories=[range(5)], sparse_output=False)

    train_onehot_labels = encoder.fit_transform(train_labels_2d)
    test_onehot_labels = encoder.transform(test_labels_2d)

    return train_texts, test_texts, train_onehot_labels, test_onehot_labels

### kobert model 가져오기
def get_kobert_model():
    model_name = 'monologg/kobert'

    tokenizer = BertTokenizer.from_pretrained(model_name)
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)

    return tokenizer, train_encodings, test_encodings

