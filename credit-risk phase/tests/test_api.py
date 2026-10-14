"""
Automated unit and integration tests for FastAPI production microservice.
"""

import pytest
from fastapi.testclient import TestClient
from src.api import app

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "documentation" in data

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["models_ready"] is True

def test_predict_german_endpoint(client):
    sample_applicant = {
        "checking_status": "A11",
        "duration": 24.0,
        "credit_history": "A32",
        "purpose": "A40",
        "credit_amount": 4000.0,
        "savings": "A61",
        "employment": "A73",
        "installment_rate": 3.0,
        "personal_status": "A93",
        "other_debtors": "A101",
        "residence_since": 2.0,
        "property": "A121",
        "age": 32.0,
        "other_installment_plans": "A143",
        "housing": "A152",
        "existing_credits": 1.0,
        "job": "A173",
        "num_dependents": 1.0,
        "telephone": "A192",
        "foreign_worker": "A201"
    }
    response = client.post("/predict/german", json=sample_applicant)
    assert response.status_code == 200
    data = response.json()
    assert data["dataset"] == "German Credit"
    assert 0.0 <= data["default_probability"] <= 1.0
    assert data["decision"] in [0, 1]
    assert "risk_level" in data
    assert data["threshold_applied"] == 0.35

def test_predict_taiwan_endpoint(client):
    sample_client = {
        "LIMIT_BAL": 120000.0,
        "SEX": "1",
        "EDUCATION": "2",
        "MARRIAGE": "1",
        "AGE": 30.0,
        "PAY_1": 2.0,
        "PAY_2": 2.0,
        "PAY_3": 0.0,
        "PAY_4": 0.0,
        "PAY_5": 0.0,
        "PAY_6": 0.0,
        "BILL_AMT1": 50000.0,
        "BILL_AMT2": 48000.0,
        "BILL_AMT3": 45000.0,
        "BILL_AMT4": 40000.0,
        "BILL_AMT5": 35000.0,
        "BILL_AMT6": 30000.0,
        "PAY_AMT1": 1000.0,
        "PAY_AMT2": 1000.0,
        "PAY_AMT3": 1000.0,
        "PAY_AMT4": 1000.0,
        "PAY_AMT5": 1000.0,
        "PAY_AMT6": 1000.0
    }
    response = client.post("/predict/taiwan", json=sample_client)
    assert response.status_code == 200
    data = response.json()
    assert data["dataset"] == "Taiwan Credit Card"
    assert 0.0 <= data["default_probability"] <= 1.0
    assert data["decision"] in [0, 1]
    assert data["threshold_applied"] == 0.46

def test_metrics_verdicts_endpoint(client):
    response = client.get("/metrics/verdicts")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
