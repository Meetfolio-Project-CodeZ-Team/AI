import redis
import json
from core.config import settings

model_key = "active_model"

rd = redis.StrictRedis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, db=0, password=settings.REDIS_PASSWORD)

def set_active_model(id:int, name:str, version:float, path:str):
    model_dict = {"model_id":id, "name":name, "version":str(version), "file_path":path}
    model_dict_json = json.dumps(model_dict, ensure_ascii=False).encode('utf-8')
    rd.set(model_key,model_dict_json)

def get_active_model():
    model_dict_json = rd.get(model_key).decode('utf-8')
    model_dict = dict(json.loads(model_dict_json))
    return model_dict