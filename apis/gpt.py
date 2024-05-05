import re
from sqlalchemy.orm import Session
from domains.model import Model
from core.config import settings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, AIMessagePromptTemplate, HumanMessagePromptTemplate

# GPT를 통한 자기소개서 첨삭
def feedback_coverletter(job, keyword, content):

  chatgpt = ChatOpenAI(model_name="gpt-3.5-turbo-1106", openai_api_key=settings.GPT_KEY, temperature=0.8)
  
  # System Prompt
  solution_template = '''
      당신은 IT 기업의 전문가로, {job} 개발자의 자기소개서를 분석하여 첨삭해주는 역할을 맡고 있습니다.
      다양한 지원자들의 자기소개서를 검토하고 현재 IT 기술에 대한 정확한 이해를 하고 있습니다.
      이를 바탕으로 지원자의 {keyword} 역량을 강조하여, 합격할 수 있도록 자기소개서를 재작성하는 역할만 수행합니다.
      당신이 작성해야 하는 자기소개서는 아래와 같습니다.

      - 아래
      {content} json
  '''
  system_prompt = SystemMessagePromptTemplate.from_template(solution_template)

  # Human Prompt
  human_template = '{content} {job} {keyword}'.format(job='{job}', keyword='{keyword}', content='{content}')
  human_prompt = HumanMessagePromptTemplate.from_template(human_template)

  # chat_prompt 설정
  chat_prompt = ChatPromptTemplate.from_messages(
      [
          system_prompt,
          human_prompt
      ]
  )

  result = chatgpt(chat_prompt.format_prompt(job=job, keyword=keyword, content=content).to_messages())

  return result.content

# GPT를 통한 자기소개서 추천 문항
def recommend_title(job, content):

  chatgpt = ChatOpenAI(model_name="gpt-3.5-turbo-1106", openai_api_key=settings.GPT_KEY, temperature=0.8)
  # System Prompt
  solution_template = '''
      당신은 IT 기업의 전문가로, {job} 엔지니어 자기소개서를 분석하여 내용에 맞는 자기소개서 작성 문항을 추천해주는 역할을 맡고 있습니다.
      다양한 지원자의 자기소개서를 검토하며, 현재의 IT 기술에 대한 정확한 이해를 바탕으로 지원자의 글에 알맞는 자기소개서 문항을 3가지 추천해주는 역할을 수행합니다.
      당신이 자기소개서 항목을 추천해야 할 자기소개서 내용은 아래와 같습니다. 예시를 참고하여 자기소개서 작성 문항을 추천해줘.

      - 아래
      {content} 

      - 예시
      {example} json
  '''

  # System Prompt
  system_prompt = SystemMessagePromptTemplate.from_template(solution_template)

  # Human Prompt
  human_template = '{job} {content}'.format(job='{job}', content='{content}')
  human_prompt = HumanMessagePromptTemplate.from_template(human_template)

  # AI Prompt
  ai_template = '{example}'.format(example='{example}')
  ai_prompt = AIMessagePromptTemplate.from_template(ai_template)

  # chat_prompt 설정
  chat_prompt = ChatPromptTemplate.from_messages(
      [
          system_prompt,
          human_prompt,
          ai_prompt
      ]
  )

  result = chatgpt(chat_prompt.format_prompt(job=job, content=content, 
                                             example='''1. 분야/직무에 대한 지원 동기'와 '해당 분야/직무를 잘 수행하기 위해 어떤 준비'를 해왔는지 구체적으로 설명해주세요. 
                                             2. 본인이 끝까지 파고들어 본 가장 의미있었던 개발 경험 또는 개발 활동에 대해 얘기해 주세요. 
                                             3. 본인의 SW개발 활동 경험 중 다른 사람과 함께 같은 목표를 위해 노력한 경험 또는 어려운 기술적 문제를 해결한 경험에 대해 얘기해 주세요. 
                                             ### \n 위와 같은 예시의 형식으로 자기소개서 내용과 알맞는 문항을 3가지 추천해줘.''').to_messages())

  return result.content

# 첨삭과 추천 문항 한 번에 처리
def gpt_feedback(job, keyword, content):
  result = {}

  content = feedback_coverletter(job, keyword, content)
  result['feedback'] = remove_special_characters(content)

  recommend = recommend_title(job, content)
  result['recommend'] = extract_texts(recommend)

  return result

# 정규 표현식
def remove_special_characters(text):
    cleaned_text = re.sub(r'[^\w\d.\sㄱ-ㅣ가-힣]', '', text)
    cleaned_text = cleaned_text.replace('\n', '')
    return cleaned_text

# 추천 문항 파싱
def extract_texts(text):
    matches = re.findall(r'(\d+)\.\s+(.*?)(?=\d+\.\s|$)', text, re.DOTALL)

    # 추출된 문장을 리스트에 저장
    extracted_texts = []
    for match in matches:
        num, text = match
        text = re.sub(r'[\"/\\\n]', '', text)
        extracted_texts.append(text)

    return extracted_texts

## ==================== ##

# kobert 두드러진 역량 분석

def analysis_skill_keyword(content):

  chatgpt = ChatOpenAI(model_name="gpt-3.5-turbo-1106", openai_api_key=settings.GPT_KEY, temperature=0.8)

  # System Prompt
  analysis_template = '''
      당신은 IT 기업의 전문가로, 채용 관련 업무에 종사하는 전문가로서, 자기소개서를 분석하여 지원자의 역량을 추출하는 역할을 맡고 있습니다.
      다양한 자기소개서를 검토하며, 현재 IT 기술에 대한 깊은 이해를 바탕으로 지원자의 뛰어난 역량을 15글자 이하로 3개 추출해야 합니다.
      아래는 지원자의 자기소개서 내용입니다.

      - 아래
      {content} .json
  '''
  system_prompt = SystemMessagePromptTemplate.from_template(analysis_template)

  # Human Prompt
  human_template = '{content}'.format(content='{content}')
  human_prompt = HumanMessagePromptTemplate.from_template(human_template)

  # chat_prompt 설정
  chat_prompt = ChatPromptTemplate.from_messages(
      [
          system_prompt,
          human_prompt
      ]
  )
  result = chatgpt(chat_prompt.format_prompt(content=content).to_messages())

  skill_keyword = extract_texts(result.content)

  return skill_keyword