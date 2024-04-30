from db.session import SessionLocal

# get_db() -> next()로 생성 후 사용
def get_db():
	db = SessionLocal()
	try:
		yield db
	finally:
		db.close()