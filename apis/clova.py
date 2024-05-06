from core.config import settings
import requests, json

class ClovaSummarizer:
  def __init__(self):
    self.client_id = settings.CLOVA_ID
    self.client_secret = settings.CLOVA_SECRET
    self.headers = {
        "X-NCP-APIGW-API-KEY-ID": self.client_id,
        "X-NCP-APIGW-API-KEY": self.client_secret,
        "Content-Type": "application/json"
    }
    self.url= settings.CLOVA_URL
    self.language = "ko" # (ko, ja)
    self.model = "general"  # (general, news)
    self.tone = "2"      # (0, 1, 2, 3)
    self.summaryCount = "3"
    
  def summarize_text(self, job_keyword, content):

    data = {
        "document": {
            "title": job_keyword + '합격 자기소개서',
            "content": content
        },
        "option": {
            "language": self.language,
            "model": self.model,
            "tone": self.tone,
            "summaryCount" : self.summaryCount
        }
    }
    response = requests.post(self.url, data=json.dumps(data), headers=self.headers)
    rescode = response.status_code

    if rescode == 200:
      return self._parse_response(response)
    else:
      print("Error : " + response.text)

  def _parse_response(self, response):
    result = json.loads(response.text)
    results = result['summary']
    result_list = [text.replace('\n', '') for text in results.split(".")]
    concat_result = ' '.join(result_list)
    return concat_result