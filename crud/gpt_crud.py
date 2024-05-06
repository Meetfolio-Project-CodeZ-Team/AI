from sqlalchemy.orm import Session
from domains.coverletter import Coverletter
from datetime import datetime
from domains.feedback import Feedback

def get_coverletter(db: Session, coverletter_id: int):

  return db.query(Coverletter).filter(Coverletter.cover_letter_id == coverletter_id).one()

def save_feedback(db: Session, cover_letter_id, response):
  feedback = Feedback(
        cover_letter_id=cover_letter_id,
        correction=response['feedback'],
        recommend_question_1=response['recommend'][0],
        recommend_question_2=response['recommend'][1],
        recommend_question_3=response['recommend'][2],
        created_at=datetime.now(),
        updated_at=datetime.now()
  )
  db.add(feedback)
  db.commit()

  return 1