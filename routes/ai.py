import os
from flask import jsonify, json, request
from db.connection import get_db
from crud.gpt_crud import get_coverletter, save_feedback, check_feedback
from apis.gpt import gpt_feedback, analysis_skill_keyword
from crud.kobert_crud import get_inactive_dataset, get_active_model_path, save_model, save_analysis, check_analysis
from apis.kobert import DataPreprocessor, DataLoaderBuilder, CustomDataset, KobertClassifier, ModelManager
from apis.clova import ClovaSummarizer
from crud.model_crud import change_model_status, soft_delete_model, get_model
from core.config import settings
from flask_restx import Namespace, Resource, Api, fields
from transformers import AutoTokenizer
from datetime import datetime
from db.redis import set_active_model, get_active_model, get_version_info, set_version_info
import re

ai = Namespace("ai", description="AI 자기소개서 피드백 및 직무 역량 분석 API")

analysis_response = ai.model('Analysis Response DTO', {
  "analysis_id": fields.Integer(description="AI 직무 역량 분석, Analysis 아이디"),
  "job_suitability": fields.Float(description="AI 직무 역량 분석 결과"),
  "skill_keywords": fields.List(fields.String(description='사용자 두드러진 역량 키워드 리스트')),
  "job_keyword": fields.String(description="사용자 지원 직무")
})
feedback_response = ai.model('Feedback Response DTO', {
  "feedback_id": fields.Integer(description="AI 피드백, Feedback 아이디"),
  "feedback": fields.String(description="AI 자기소개서 피드백 결과"),
  "recommend": fields.List(fields.String(description='AI 자기소개서 추천 문항 리스트'))
})
train_response = ai.model('Model Train Response DTO', {
  "model_id": fields.Integer(description="모델 아이디"),
  "created_at": fields.DateTime(description="요청 응답 시간")
})
model_change_response = ai.model('Model Change Response DTO',{
  "model_id" : fields.Integer(description="모델 아이디"),
  "name": fields.String(description="모델명"),
  "version": fields.String(description="버전"),
  "file_path": fields.String(description="파일 경로")
})
delete_response = ai.model('Model Delete Response DTO', {
  "model_id": fields.Integer(description="모델 아이디"),
  "created_at": fields.DateTime(description="요청 응답 시간")
})

@ai.route("/coverLetter-analysis/<int:cover_letter_id>")
class Analysis(Resource):
  @ai.response(200, 'Success', analysis_response)
  def post(self, cover_letter_id):
    """Kobert를 통한 AI 직무 역량 분석 API"""
    # data = request.json
    db = get_db()
    session = next(db)

    # 존재하는 직무 역량 분석인지 확인
    check_analysis(session, cover_letter_id)

    result = get_coverletter(session, cover_letter_id)
    data = {"cover_letter_id": result.cover_letter_id,
            "answer": result.answer,
            "job_keyword": result.job_keyword}

    data_loadbuilder = DataLoaderBuilder()
    tokenizer = data_loadbuilder.get_tokenizer()

    model_path = get_active_model_path()
    model_manager = ModelManager()
    model = model_manager.get_model(model_path)

    # tokenizer = AutoTokenizer.from_pretrained("monologg/kobert")
    
    kobert_model = KobertClassifier(model, tokenizer)

    clova = ClovaSummarizer()
    summary_data = clova.summarize_text(data['job_keyword'], data['answer'])

    outputs = kobert_model.predict(tokenizer, summary_data)
    proba = kobert_model.predict_proba(outputs, data['job_keyword'])

    skill_keywords = analysis_skill_keyword(data['answer'])
    analysis_id = save_analysis(session, cover_letter_id, proba, skill_keywords)

    return {"analysis_id": analysis_id, "job_suitability": round(proba * 100), 
            "skill_keywords": skill_keywords, "job_keyword": data['job_keyword']}


@ai.route("/coverLetter-feedbacks/<int:cover_letter_id>")
class Feedback(Resource):
  @ai.response(200, 'Success', feedback_response)
  def post(self, cover_letter_id):
    """GPT를 통한 AI 자기소개서 피드백 API"""
    db = get_db()
    session = next(db)

    # 존재하는 피드백인지 확인
    check_feedback(session, cover_letter_id)
    
    # 자기소개서 조회
    result = get_coverletter(session, cover_letter_id)
    
    data = {"cover_letter_id": result.cover_letter_id,
            "answer": result.answer,
            "keyword1": result.keyword_1,
            "keyword2": result.keyword_2,
            "job_keyword": result.job_keyword}
    
    keyword = data["keyword1"] + "," + data["keyword2"]
    response = gpt_feedback(data["job_keyword"], keyword, data["answer"])

    print(response)

    # 피드백 저장
    feedback_id = save_feedback(session, cover_letter_id, response)
    response["feedback_id"] = feedback_id

    return jsonify(response)

@ai.route("/admins/model-management/train")
class ModelTrain(Resource):
  @ai.response(200, 'Success', train_response)
  def post(self):
    """Kobert 모델 추가 학습 API"""

    # 추가학습 초과 알림
    active_model_version = get_active_model()["version"]
    pattern = r"(\d+\.\d+)\.\d+"
    match = re.match(pattern, active_model_version)
    version = match.group(1)
    version_info = get_version_info(version)
    trained_count = version_info["trained_count"]
    if trained_count+1 > 9: 
      abort(400, description = "학습할 수 있는 횟수를 초과하였습니다. 관리자에게 문의 바랍니다.")
    else:
      # 0. DB 초기화
      db = get_db()
      session = next(db)

      # 0. Class 초기화
      data_preprocessor = DataPreprocessor()
      data_loadbuilder = DataLoaderBuilder()
      
      # 1. 학습할 데이터 가져오기
      datasets, dataset_list = get_inactive_dataset(session)
      train_texts, test_texts, train_labels, test_labels = data_preprocessor.preprocess_data(dataset_list)

      train_encodings = data_loadbuilder.get_encoding(train_texts)
      test_encodings = data_loadbuilder.get_encoding(test_texts)
      kobert_tokenizer = data_loadbuilder.get_tokenizer()

      train_loader = data_loadbuilder.get_dataloader(train_encodings, train_labels)
      test_loader = data_loadbuilder.get_dataloader(test_encodings, test_labels)

      # 2. 모델 불러오기
      model_path = get_active_model_path()
      model_manager = ModelManager()
      model = model_manager.get_model(model_path)
      
      # 3. 모델 초기화
      kobert_model = KobertClassifier(model, kobert_tokenizer)

      # 4. 모델 추가 학습
      average_loss = kobert_model.train(train_loader)
      accuracy = kobert_model.evaluate(train_loader)

      # 5. 새 모델 저장
      new_model, new_version = model_manager.save_model(model, model_path)
      saved_model = save_model(session, new_model, new_version, accuracy, average_loss)
      set_version_info(version, new_version, trained_count+1)

      # 6. 가져온 데이터셋 -> ACTIVE
      for dataset in datasets:
        dataset.status = 'ACTIVE'
      session.commit()

      return {"model_id": saved_model.model_id, "created_at": datetime.now().isoformat()}

@ai.route("/admins/model-management/version/<int:model_id>")
class VesionControl(Resource):
  @ai.response(200, 'Success', model_change_response)
  def post(self,model_id ):
    """모델 변경 API"""
    from_model = get_active_model()
    from_model_id = from_model["model_id"]
    # MYSQL DB 반영
    db = get_db()
    session = next(db)
    to_model = change_model_status(session, model_id, from_model_id)
    # REDIS DB 반영
    set_active_model(to_model.model_id, to_model.name, to_model.version, to_model.file_path)
    changed_model = get_active_model()
    return changed_model

@ai.route("/developer/model/<int:model_id>")
class ModelDelete(Resource):
  @ai.response(200, 'Success', model_change_response)
  def delete(self, model_id):
    """개발자 모델(파일) 삭제 API"""
    db = get_db()
    session = next(db)
    file_path = get_model(session, model_id).file_path
    if os.path.exists(file_path):
      os.remove(file_path)
    soft_delete_model(session, model_id)
    return {"model_id": model_id, "deleted_at": datetime.now().isoformat()}