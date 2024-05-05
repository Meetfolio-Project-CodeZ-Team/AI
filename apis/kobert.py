import pandas as pd
from konlpy.tag import Okt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
import torch.nn.functional as F
from core.config import settings

### Parameter
max_length = 400
batch_size = 16
warmup_ratio = 0.1
num_epochs = 15
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
    train_texts, test_texts, train_onehot_labels, test_onehot_labels = self._train_test_split(df)
    return train_texts, test_texts, train_onehot_labels, test_onehot_labels

  def _process_job_label(self, df):
    label_mapping = {"백엔드": 0, "웹개발": 1, "앱개발": 2, "AI": 3, "디자인": 4}
    df['label'] = df['job'].map(label_mapping)
    df.drop(columns=['job'], inplace=True)
    return df

  def _process_korean_text(self, df):
    okt = Okt()
    df['data'] = df['data'].map(lambda x: ' '.join(okt.morphs(x, stem=True)))
    return df

  def _train_test_split(self, df):
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

class DataLoaderBuilder:
  def __init__(self):
    self.tokenizer = BertTokenizer.from_pretrained('monologg/kobert')

  def get_encoding(self, texts):
    encodings = self.tokenizer(texts, truncation=True, padding=True, max_length=max_length)
    return encodings

  def get_dataloader(self, encodings, labels):
    dataset = CustomDataset(encodings, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
  
  def get_tokenizer(self):
    return self.tokenizer

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


class KobertClassifier:

  def __init__(self, model, tokenizer):
    # self.model = BertForSequenceClassification.from_pretrained('monologg/kobert', num_labels=5)
    self.model = model
    self.tokenizer = tokenizer
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # model train
  def train(self, train_loader):
    device = self.device
    optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
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
    device = self.device
    self.model.eval()
    correct_predictions = 0
    total_predictions = 0 

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
  def predict(self, tokenizer, input_text):
    device = self.device
    # 입력 테스트를 tokenizer를 사용하여 인코딩, 반환된 텐서를 input_endcoding에 저장
    input_encoding = tokenizer.encode_plus(
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

    print(f"Predicted_Labels: {predicted_labels}")

    return outputs

  def predict_proba(self, outputs, job):
    # 예측된 로짓 값을 소프트맥스 함수를 통해 확률로 변환
    predicted_probs = F.softmax(outputs.logits, dim=1)
    label_mapping = {"BACKEND": 0, "WEB": 1, "APP": 2, "DESIGN": 3, "AI": 4}
    job = label_mapping[job]

    proba = predicted_probs[0][job].item()
    print(f"사용자가 입력한 직무에 대한 역량 분석 : {proba}")

    return proba

class ModelManager:

  def __init__(self):
    self.path = settings.MODEL_PATH
    self.model_name = settings.MODEL_NAME
    self.kobert_default = settings.KOBERT_DEFAULT 

  def save_model(self, model, past_file_name):
    new_version = get_version(past_file_name)
    new_file_name = self.model_name + new_version + '_state_dict.pt'
    torch.save(model.state_dict(), new_file_name)

    return new_file_name, new_version

  def get_model(self, model_state_dict):
    model = torch.load(self.kobert_default)
    # model.load_state_dict(torch.load(model_state_dict))

    return model
  
  def next_version(past_file_name):
    pattern = r"meetfolio_model_v([\d.]+)_state_dict.pt"
    match = re.search(pattern, past_file_name)
    if match:
      version_str = match.group(1)
      version = float(version_str)
      new_version = version + 0.1
      return new_version
    else:
      return 1