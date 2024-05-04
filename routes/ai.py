from flask import Blueprint, jsonify, json
from db.connection import get_db
from crud.gpt_crud import get_coverletter, save_feedback
from apis.gpt import gpt_feedback
from crud.kobert_crud import get_inactive_dataset, get_active_model
from apis.kobert import DataPreprocessor, DataLoaderBuilder, CustomDataset, KobertClassifier, ModelManager
from apis.clova import ClovaSummarizer
from core.config import settings

# 객체 이름 : 'meetAI' / @RequestMapping : url_prefix 
bp = Blueprint('meetAI', __name__, url_prefix="/api")

@bp.route("/test")
def test():
  db = get_db()
  session = next(db)
  
  model_path = get_active_model(session)
  return model_path

@bp.route("/coverLetter-analysis/{coverLetterId}", methods=['POST'])
def analysis():
  return {"message": "hello fastapi"}

@bp.route("/coverLetter-feedbacks/<int:cover_letter_id>", methods=['POST'])
def feedback(cover_letter_id):

  db = get_db()
  session = next(db)

  # 자기소개서 조회
  result = get_coverletter(session, cover_letter_id)
  
  data = {"cover_letter_id": result.cover_letter_id,
          "answer": result.answer,
          "keyword1": result.keyword_1,
          "keyword2": result.keyword_2,
          "job_keyword": result.job_keyword}
  
  keyword = data["keyword1"] + "," + data["keyword2"]
  response = gpt_feedback(data['job_keyword'], keyword, data['answer'])

  # 피드백 저장
  save_feedback(session, cover_letter_id, response)

  return jsonify(response)

@bp.route("/admins/model-management/train", methods=['POST'])
def model_train():
  
  # DB 초기화
  db = get_db()
  session = next(db)

  # Class 초기화
  data_preprocessor = DataPreprocessor()
  data_loadbuilder = DataLoaderBuilder()
  
  # 학습할 데이터 가져오기
  dataset = get_inactive_dataset(session)
  train_texts, test_texts, train_labels, test_labels = data_preprocessor.preprocess_data(dataset)

  train_encodings = data_loadbuilder.get_encoding(train_texts)
  test_encodings = data_loadbuilder.get_encoding(test_texts)
  kobert_tokenizer = data_loadbuilder.get_tokenizer()

  train_loader = data_loadbuilder.get_dataloader(train_encodings, train_labels)
  test_loader = data_loadbuilder.get_dataloader(test_encodings, test_labels)

  # TODO : 파인튜닝한 모델로 변경
  model_path = settings.KOBERT_MODEL
  model_manager = ModelManager(model_path)
  model = model_manager.get_model()
  
  kobert_model = KobertClassifier(kobert_tokenizer, model)

  average_loss = kobert_model.train(train_loader)
  accuracy = kobert_model.evaluate(train_loader)

  response = {"average_loss": round(average_loss, 4), "accuracy": round(accuracy, 4)}
  return response


@bp.route("/admins/model-management/version/{modelId}", methods=['POST'])
def model_change():
  return 'hello test'