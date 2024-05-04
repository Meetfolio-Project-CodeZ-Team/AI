import pandas as pd
from konlpy.tag import Okt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
import torch.nn.functional as F
import requests
import json

### Parameter
max_len = 400
batch_size = 16
warmup_ratio = 0.1
num_epochs = 10
max_grad_norm = 1
log_interval = 200
learning_rate =  5e-5
### ===================================== ###

class DataPreprocessor:
  # 추가 학습할 데이터 설정
  def preprocess_data(self, datasets):
    df = pd.DataFrame(datasets)
    df = self._process_job_label(df)
    df = self._process_korean_text(df)
    return df

  def _process_job_label(self, df):
    label_mapping = {"백엔드": 0, "웹개발": 1, "앱개발": 2, "AI": 3, "디자인": 4}
    df['label'] = df['job'].map(label_mapping)
    df.drop(columns=['job'], inplace=True)
    return df

  def _process_korean_text(self, df):
    okt = Okt()
    df['data'] = df['data'].map(lambda x: ' '.join(okt.morphs(x, stem=True)))
    return df

class DataLoaderBuilder:
  def __init__(self, batch_size):
    self.batch_size = batch_size
    self.model_name = 'monologg/kobert'
    self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
    self.train_encodings = None
    self.test_encodings = None

  def get_tokenizer_encoding(self):
    self.train_encodings = self.tokenizer(train_texts, truncation=True, padding=True)
    self.test_encodings = self.tokenizer(test_texts, truncation=True, padding=True)

    return train_encodings, test_encodings

  def get_dataloader(self, encodings, labels):
    dataset = CustomDataset(encodings, labels)

    return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

# Kobert 모델에 학습 시킬 Dataset 만들기
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# train_dataset = CustomDataset(train_encodings, train_onehot_labels)
# test_dataset = CustomDataset(test_encodings, test_onehot_labels)
  

class KobertClassifier:

  def __init__(self, tokenizer):
    self.model = BertForSequenceClassification.from_pretrained('monologg/kobert', num_labels=5)
    self.tokenizer = tokenizer

  # 훈련 & 테스트 데이터 분리
  def data_train_test_split(self, df):
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
  
  # # DataLoader 가져오기
  # def get_train_test_dataloader(train_dataset, test_dataset):
  #     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  #     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

  #     return train_loader, test_loader

  # model train
  def train(self, train_loader):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    self.model.to(device)

    for epoch in range(num_epochs):
      self.model.train()
      total_loss = 0

      for batch in tqdm(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # 이전 배치에서의 그래디언트 초기화
        optimizer.zero_grad()

        # 모델에 인풋을 전달하여 출력 계산
        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)

        # 모델의 출력에서 손실 추출
        loss = outputs.loss

        # 배치 내의 손실을 총 손실에 더하기
        total_loss += loss.item()

        # 역전파를 통해 그래디언트 계산
        loss.backward()

        # 계산된 그래디언트로 파라미터 업데이트
        optimizer.step()

    average_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {average_loss:.4f}")

  return average_loss

  # model test
  def evaluate(self, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.model.eval() # 모델을 평가 모드로 전환
    correct_predictions = 0 # 올바르게 예측된 샘플의 수
    total_predictions = 0   # 전체 예측한 샘플의 수

    with torch.no_grad():
      for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = self.model(input_ids, attention_mask=attention_mask)
        _, predicted_labels = torch.max(outputs.logits, dim=1)
        labels = torch.argmax(labels, dim=1)

        correct_predictions += torch.sum(predicted_labels == labels).item()
        total_predictions += labels.size(0)

    accuracy = correct_predictions / total_predictions
    print(f"Test Accuracy: {accuracy:.4f}")

    return accuracy


  # predict model
  def predict(self, input_text):
      # 입력 테스트를 tokenizer를 사용하여 인코딩, 반환된 텐서를 input_endcoding에 저장
      input_encoding = self.tokenizer.encode_plus(
          input_text,
          truncation=True,
          padding=True,
          return_tensors='pt'
      )

      input_ids = input_encoding['input_ids'].to(device)
      attention_mask = input_encoding['attention_mask'].to(device)

      self.model.eval()
      with torch.no_grad():
          # 모델에 입력과 어텐션 마스크를 전달하여 출력을 계산
          outputs = self.model(input_ids, attention_mask=attention_mask)
          _, predicted_labels = torch.max(outputs.logits, dim=1)
      predicted_labels = predicted_labels.item()

      print(predicted_labels)

      return outputs

  def predict_proba(self, outputs, job):
      # 예측된 로짓 값을 소프트맥스 함수를 통해 확률로 변환
      predicted_probs = F.softmax(outputs.logits, dim=1)
      reverse_mapping = {0: "백엔드", 1: "웹개발", 2: "앱개발", 3: "AI", 4: "디자인"}

      # 가장 높은 확률을 가진 클래스의 인덱스를 예측 레이블로 선택
      predicted_label_index = torch.argmax(predicted_probs, dim=1)
      predicted_probability = predicted_probs[0][predicted_label_index]

      print("사용자가 입력한 직무에 대한 역량 분석")
      proba = predicted_probs[0][job].item()

      return proba

  def save_model(path):
      torch.save(self.model, path + '모델명.pt')
      torch.save(self.model.state_dict(), '모델명_state_dict.pt')

