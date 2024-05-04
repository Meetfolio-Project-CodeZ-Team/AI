from flask import Blueprint
from apis.gpt import gpt_feedback

# 객체 이름 : 'test' / @RequestMapping : url_prefix 
bp = Blueprint('test', __name__, url_prefix="/")

@bp.route("/")
def root():
  return {"message": "hello fastapi"}

@bp.route("/test")
def hello_pybo():
  return gpt_feedback()