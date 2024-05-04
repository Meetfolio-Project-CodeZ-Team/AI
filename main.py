from flask import Flask, jsonify
from routes import test, ai
from db.connection import get_db
from apis.gpt import test_index

def create_app():
  app =Flask(__name__)

  app.register_blueprint(test.bp)
  app.register_blueprint(ai.bp)

  @app.route("/abc")
  def db_test():

    db = get_db()
    session = next(db)
    # data = session.query(Model).all()
    data = test_index(session)
    # 데이터를 JSON 형식으로 변환
    results = jsonify([{"model_id": model.model_id,
                              "accuracy": model.accuracy,
                              "loss": model.loss,
                              "version": model.version,
                              "activated_date": model.activated_date,
                              "created_at": model.created_at,
                              "learned_date": model.learned_date,
                              "updated_at": model.updated_at,
                              "file_name": model.file_name,
                              "file_path": model.file_path,
                              "name": model.name,
                              "status": model.status} for model in data])
    return results

    
  return app