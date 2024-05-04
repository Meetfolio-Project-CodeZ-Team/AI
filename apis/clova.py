from core.config import settings

def set_clova():
  client_id = settings.CLOVA_ID
  client_secret = settings.CLOVA_SECRET
  headers = {
      "X-NCP-APIGW-API-KEY-ID": client_id,
      "X-NCP-APIGW-API-KEY": client_secret,
      "Content-Type": "application/json"
  }
  url= settings.CLOVA_URL

  language = "ko" # (ko, ja)
  model = "general"  # (general, news)
  tone = "2"      # (0, 1, 2, 3)
  summaryCount = "3"

  return client_id, client_secret, headers, url, language, model, tone, summaryCount
  

def str_to_json(response):
    result = json.loads(response.text)
    return result['summary']

title= "백엔드 합격 자기소개서"
content = '''다양한 공부를 통해 개발자로서의 기초를 다진 후, 제가 배운 것들을 확인해 보는 동시에 여러 문제를 경험하며 이를 해결해 보기 위해 개인 프로젝트를 진행하였습니다.
Spring boot환경에서 Spring data JPA와 H2 Database, spring security, 그리고 이를 보여주는 프론트 뷰를 구현하여 보았고, AWS상에서 구동시켜 확인해 보았습니다.
개발의 시작부터 끝까지 혼자 진행해 보면서 다양한 문제를 실제로 접할 수 있었고, 이전에 배웠던 내용을 실제로 적용하거나 새로운 기술들 시도해 보면서 조금 더 개발에의 흥미를 가질 수 있었습니다.
또한 단순히 기능의 구현에 초점을 맞추어 개발하는 것 보다는, 최선의 퍼포먼스와 깔끔한 코드 작성을 위해 계속해서 노력해야 한다는 사실을 알 수 있었습니다.'''

def post_clova(title, content, language, model, tone, summaryCount):

  data = {
      "document": {
          "title": title,
          "content" : content
      },
      "option": {
          "language": language,
          "model": model,
          "tone": tone,
          "summaryCount" : summaryCount
      }
  }
  response = requests.post(url, data=json.dumps(data), headers=headers)
  rescode = response.status_code

  if(rescode == 200):
      results = str_to_json(response)
      result_list = [text.replace('\n', '') for text in results.split(".")]
  else:
      print("Error : " + response.text)