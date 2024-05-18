from flask import Flask, jsonify, request
from routes.ai import ai
from db.connection import get_db
from flask_restx import Api, Resource
from flask_cors import CORS
from db.redis import set_active_model,get_active_model, set_version_info, get_version_info
import re

def create_app():

  app =Flask(__name__)
  CORS(app, resources={r'*': {'origins': ['http://www.meetfolio.kro.kr', 'http://localhost:3000', 'http://www.meetfolio.kro.kr:60005']}})
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
      return get_active_model()
    
  return app