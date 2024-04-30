from flask import Blueprint

# 객체 이름 : 'meetAI' / @RequestMapping : url_prefix 
bp = Blueprint('meetAI', __name__, url_prefix="/api")

@bp.route("/coverLetter-analysis/{coverLetterId}", methods=['POST'])
def analysis():
  return {"message": "hello fastapi"}

@bp.route("/coverLetter-feedbacks/{coverLetterId}", methods=['POST'])
def feedback():
  return 'hello test'

@bp.route("/admins/model-management/train", methods=['POST'])
def model_train():
  return 'hello test'

@bp.route("/admins/model-management/version/{modelId}", methods=['POST'])
def model_change():
  return 'hello test'