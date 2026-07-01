from fastapi import FastAPI, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from sqlalchemy import Column, String, JSON, DateTime
from pydantic import BaseModel
from datetime import datetime
import uuid
import os

import database
from generate_synthetic_teleop import generate_dataset
from quality_gate import DataQualityGate

# Database Models
class ResponseModel(database.Base):
    __tablename__ = "responses"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    interviewee_name = Column(String, nullable=True)
    company = Column(String, nullable=True)
    answers = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

# Create tables in PostgreSQL (or SQLite locally)
database.Base.metadata.create_all(bind=database.engine)

app = FastAPI(title="Verifier Node API")

# Mount the static directory to serve HTML/CSS/JS
app.mount("/static", StaticFiles(directory="static"), name="static")

# Pydantic schema for incoming requests
class QuestionnaireSubmission(BaseModel):
    interviewee_name: str
    company: str
    answers: dict

@app.post("/api/responses")
def submit_response(submission: QuestionnaireSubmission, db: Session = Depends(database.get_db)):
    db_response = ResponseModel(
        interviewee_name=submission.interviewee_name,
        company=submission.company,
        answers=submission.answers
    )
    db.add(db_response)
    db.commit()
    db.refresh(db_response)
    return {"status": "success", "id": db_response.id}

@app.get("/")
def read_root():
    return RedirectResponse(url="/static/index.html")

@app.get("/questionnaire")
def serve_questionnaire():
    return RedirectResponse(url="/static/questionnaire.html")

@app.get("/diagnostic")
def serve_diagnostic():
    return RedirectResponse(url="/static/diagnostic.html")

from generate_robomimic_mock import create_robomimic_mock
from quality_gate_raw import EdgeComputeQualityGate

@app.get("/api/diagnostic-demo")
def run_diagnostic_demo():
    """Runs the Data Quality Gate on a raw Robomimic HDF5 dataset and returns the report & plot."""
    dataset_file = "static/robomimic_mh_raw.hdf5"
    
    # Generate the dataset if it doesn't exist in the container
    if not os.path.exists(dataset_file):
        create_robomimic_mock(dataset_file)
        
    gate = EdgeComputeQualityGate(dataset_file)
    report = gate.analyze()
    
    return {"status": "success", "report": report}
