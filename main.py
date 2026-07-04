import re
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from sqlalchemy import Column, String, JSON, DateTime
from pydantic import BaseModel
from datetime import datetime, timezone
import uuid
import secrets
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
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

# Create tables in PostgreSQL (or SQLite locally)
database.Base.metadata.create_all(bind=database.engine)

app = FastAPI(title="Verifier Node API")

# Mount the static directory to serve HTML/CSS/JS
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Security: API key auth for diagnostic endpoints ---
API_KEY = os.environ.get("API_KEY", "")

def verify_api_key(request: Request):
    """
    Checks X-API-Key header on protected endpoints.
    """
    if not API_KEY:
        raise HTTPException(status_code=500, detail="Server misconfiguration: API_KEY not set")
    key = request.headers.get("X-API-Key", "")
    if not secrets.compare_digest(key, API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# --- Security: SSRF-safe Slack webhook validation ---
SLACK_WEBHOOK_PATTERN = re.compile(r"^https://hooks\.slack\.com/services/[A-Za-z0-9/]+$")

def validate_slack_webhook(webhook: str = None) -> str:
    """
    Returns validated webhook URL or None.
    Only allows official Slack webhook URLs to prevent SSRF.
    """
    if not webhook:
        return None
    if not SLACK_WEBHOOK_PATTERN.match(webhook):
        raise HTTPException(
            status_code=400,
            detail="Invalid slack_webhook URL. Only https://hooks.slack.com/services/* is allowed."
        )
    return webhook


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

from generate_lerobot_buffer import create_lerobot_buffer_mock
from lerobot_buffer_hook import ArchitectureAwareDriftGate

@app.get("/api/diagnostic-demo")
def run_diagnostic_demo(
    request: Request,
    slack_webhook: str = None,
    _auth: None = Depends(verify_api_key)
):
    """
    V3.7 Endpoint: Simulates a real-time lerobot-record buffer hook.
    Runs synchronously (FastAPI threadpool) to avoid blocking the event loop.
    """
    try:
        validated_webhook = validate_slack_webhook(slack_webhook)

        # Step 1: Generate simulated real-time buffer streams
        episodes = create_lerobot_buffer_mock()
        
        # Step 2: Run the Architecture-Aware Drift Gate on the buffer
        gate = ArchitectureAwareDriftGate(episodes, slack_webhook=validated_webhook)
        report = gate.analyze()
        
        return {"status": "success", "report": report}
        
    except HTTPException:
        raise  # Re-raise validation errors
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/diagnostic-real")
def run_diagnostic_real(
    request: Request,
    slack_webhook: str = None,
    _auth: None = Depends(verify_api_key)
):
    """
    Downloads chunk-000 from HF lerobot/aloha_mobile_cabinet and runs the quality gate on real robot teleop.
    Runs synchronously (FastAPI threadpool) to avoid blocking the event loop.
    """
    try:
        validated_webhook = validate_slack_webhook(slack_webhook)

        parquet_url = "https://huggingface.co/datasets/lerobot/aloha_mobile_cabinet/resolve/main/data/chunk-000/file-000.parquet"
        # Cache to data/ directory instead of static/ to prevent public serving
        data_dir = "data"
        os.makedirs(data_dir, exist_ok=True)
        local_path = os.path.join(data_dir, "aloha_mobile_cabinet.parquet")
        
        # Cache the dataset locally so subsequent clicks are instantaneous
        if not os.path.exists(local_path):
            import urllib.request
            print(f"Downloading {parquet_url}...")
            urllib.request.urlretrieve(parquet_url, local_path)
            print("Download complete.")
            
        gate = ArchitectureAwareDriftGate(slack_webhook=validated_webhook)
        report = gate.analyze_real_parquet(local_path)
        
        return {"status": "success", "report": report}
        
    except HTTPException:
        raise  # Re-raise validation errors
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
