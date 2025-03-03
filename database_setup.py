from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.orm import sessionmaker, declarative_base
import pandas as pd

DATABASE_URL = "sqlite:///blood_tests.db"

engine = create_engine(DATABASE_URL, echo=True)
Base = declarative_base()

class BloodTest(Base):
    __tablename__ = "blood_tests"
    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(String)
    test_name = Column(String)
    value = Column(Float)
    units = Column(String)

Base.metadata.create_all(engine)

df = pd.read_csv("mock_blood_tests.csv")

Session = sessionmaker(bind=engine)
session = Session()
for _, row in df.iterrows():
    test = BloodTest(
        patient_id=row["Patient ID"],
        test_name=row["Test Name"],
        value=row["Value"],
        units=row["Units"]
    )
    session.add(test)

session.commit()
session.close()
print ("Mock blood tests put into SQLite database (blood_tests.db")
