from sqlalchemy.orm import Session
from domains.coverletter import Coverletter
from datetime import datetime
from domains.feedback import Feedback
from sqlalchemy.orm.exc import NoResultFound
from flask import abort

def get_coverletter(db: Session, coverletter_id: int):

  # 예외 처리
  try:
    cover_letter = db.query(Coverletter).filter(Coverletter.cover_letter_id == coverletter_id).one()

    if not cover_letter.job_keyword:
      abort(400, description = "Job Keyword is missing!")
    if len(cover_letter.answer) < 200:
      abort(400, description = "자기소개서 글자수가 너무 적습니다.")
    if not cover_letter.keyword_1 or not cover_letter.keyword_2:
      abort(400, description = "Keyword1 or Keyword2 is missing!!")
    
    return cover_letter
  except NoResultFound:
    abort(400, description="Cover letter with ID {} not found.".format(coverletter_id))

def save_feedback(db: Session, cover_letter_id, response):

  recommend = response.get('recommend', [])

  if recommend:
    feedback = Feedback(
          cover_letter_id=cover_letter_id,
          correction=response['feedback'],
          recommend_question_1=response['recommend'][0],
          recommend_question_2=response['recommend'][1],
          recommend_question_3=response['recommend'][2],
          created_at=datetime.now(),
          updated_at=datetime.now()
    )
  else:
    feedback = Feedback(
          cover_letter_id=cover_letter_id,
          correction=response['feedback'],
          recommend_question_1='해당 분야/직무를 잘 수행하기 위해 어떤 준비를 해왔는지 구체적으로 설명해주세요.',
          recommend_question_2='본인이 끝까지 파고들어 본 가장 의미있었던 개발 경험 또는 개발 활동에 대해 얘기해 주세요. ',
          recommend_question_3='본인의 개발 활동 경험 중 다른 사람과 함께 같은 목표를 위해 노력한 경험에 대해 얘기해 주세요.',
          created_at=datetime.now(),
          updated_at=datetime.now()
    )

  db.add(feedback)
  db.commit()

  return feedback.feedback_id