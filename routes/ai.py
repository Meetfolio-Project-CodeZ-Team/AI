from flask import Blueprint, jsonify, json
from db.connection import get_db
from crud.gpt_crud import get_coverletter, save_feedback
from apis.gpt import gpt_feedback
from crud.kobert_crud import get_active_dataset

# 객체 이름 : 'meetAI' / @RequestMapping : url_prefix 
bp = Blueprint('meetAI', __name__, url_prefix="/api")

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
  db = get_db()
  session = next(db)
  
  return get_active_dataset(session)

@bp.route("/admins/model-management/version/{modelId}", methods=['POST'])
def model_change():
  return 'hello test'