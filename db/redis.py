import redis
import json
from core.config import settings

active_model_key = "active_model"

rd = redis.StrictRedis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, db=0, password=settings.REDIS_PASSWORD)

def set_active_model(id:int, name:str, version:str, path:str):
    model_dict = {"model_id":id, "name":name, "version":version, "file_path":path}
    model_dict_json = json.dumps(model_dict, ensure_ascii=False).encode('utf-8')
    rd.set(active_model_key,model_dict_json)

def get_active_model():
    model_dict_json = rd.get(active_model_key).decode('utf-8')
    model_dict = dict(json.loads(model_dict_json))
    return model_dict

def set_version_info(version:float, alpha_version:float, trained_count: int):
    version_key = "v" + str(version)
    version_dict = {"alpha_version" : str(alpha_version), "trained_count" :trained_count }
    version_dict_json = json.dumps(version_dict, ensure_ascii=False).encode('utf-8')
    rd.set(version_key, version_dict_json)
    
def get_version_info(version: str):
    version_key = "v" + str(version)
    version_dict_json = rd.get(version_key).decode('utf-8')
    version_dict = dict(json.loads(version_dict_json))
    return version_dict
