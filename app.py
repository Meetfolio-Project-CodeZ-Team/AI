from flask import Flask, jsonify, request
from routes.ai import ai
from db.connection import get_db
from flask_restx import Api, Resource
from flask_cors import CORS
# from crud.kobert_crud import get_active_model
from db.redis import set_active_model,get_active_model

def create_app():

  app =Flask(__name__)
  CORS(app, resources={r'*': {'origins': ['http://www.meetfolio.kro.kr', 'http://localhost:3000']}})
  # API 객체 등록
  api = Api(
    app,
    version='v1.0',
    title="Meetfolio's AI API Server",
    description="Meetfolio's Kobert, GPT, Clova API Server!",
    terms_url="/",
  )
  api.add_namespace(ai, "/api")
 
  @api.route("/test")
  class Test(Resource):
    def post(self):
      """현재 버전 확인"""
      # set_active_model(1,"meetfolio-model",1.00,"/home/t24105/v0.9src/ai/model/meetfolio_model_v1.pt")
      result = get_active_model()
      return result
    
  return app