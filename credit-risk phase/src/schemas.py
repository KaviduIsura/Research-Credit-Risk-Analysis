"""
Pydantic data schemas for API input validation and prediction responses.
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

class GermanApplicant(BaseModel):
    """Schema for German Credit Loan Applicant."""
    checking_status: str = Field(default="A11", description="Status of existing checking account (e.g. A11: < 0 DM, A12: 0-200 DM, A14: no account)")
    duration: float = Field(default=24.0, ge=1, le=100, description="Duration in months")
    credit_history: str = Field(default="A32", description="Credit history (e.g. A32: existing credits paid back duly, A34: critical account)")
    purpose: str = Field(default="A40", description="Purpose (e.g. A40: car new, A41: car used, A43: radio/television)")
    credit_amount: float = Field(default=3000.0, ge=100, description="Credit amount in Deutsche Mark")
    savings: str = Field(default="A61", description="Savings account/bonds (e.g. A61: < 100 DM, A65: no savings)")
    employment: str = Field(default="A73", description="Present employment since (e.g. A73: 1-4 years, A75: >= 7 years)")
    installment_rate: float = Field(default=3.0, ge=1, le=4, description="Installment rate in percentage of disposable income")
    personal_status: str = Field(default="A93", description="Personal status and sex (e.g. A93: male single, A92: female div/dep/mar)")
    other_debtors: str = Field(default="A101", description="Other debtors / guarantors (A101: none, A102: co-applicant, A103: guarantor)")
    residence_since: float = Field(default=2.0, ge=1, le=4, description="Present residence since in years")
    property: str = Field(default="A121", description="Property ownership (A121: real estate, A124: no property)")
    age: float = Field(default=35.0, ge=18, le=120, description="Age in years")
    other_installment_plans: str = Field(default="A143", description="Other installment plans (A141: bank, A143: none)")
    housing: str = Field(default="A152", description="Housing (A151: rent, A152: own, A153: for free)")
    existing_credits: float = Field(default=1.0, ge=1, le=10, description="Number of existing credits at this bank")
    job: str = Field(default="A173", description="Job qualification (A173: skilled employee, A174: management/self-employed)")
    num_dependents: float = Field(default=1.0, ge=1, le=5, description="Number of people being liable to provide maintenance for")
    telephone: str = Field(default="A191", description="Telephone registered under customer name (A191: none, A192: yes)")
    foreign_worker: str = Field(default="A201", description="Foreign worker status (A201: yes, A202: no)")

class TaiwanClient(BaseModel):
    """Schema for Taiwan Credit Card Client."""
    LIMIT_BAL: float = Field(default=150000.0, ge=1000, description="Amount of given credit in NT dollar")
    SEX: str = Field(default="1", description="Gender (1 = male; 2 = female)")
    EDUCATION: str = Field(default="1", description="Education (1 = graduate school; 2 = university; 3 = high school; 4 = others)")
    MARRIAGE: str = Field(default="1", description="Marital status (1 = married; 2 = single; 3 = others)")
    AGE: float = Field(default=35.0, ge=18, le=100, description="Age in years")
    PAY_1: float = Field(default=0.0, ge=-2, le=9, description="Repayment status in September (-1=pay duly, 1=delay 1 month, 2=delay 2 months)")
    PAY_2: float = Field(default=0.0, ge=-2, le=9, description="Repayment status in August")
    PAY_3: float = Field(default=0.0, ge=-2, le=9, description="Repayment status in July")
    PAY_4: float = Field(default=0.0, ge=-2, le=9, description="Repayment status in June")
    PAY_5: float = Field(default=0.0, ge=-2, le=9, description="Repayment status in May")
    PAY_6: float = Field(default=0.0, ge=-2, le=9, description="Repayment status in April")
    BILL_AMT1: float = Field(default=20000.0, description="Amount of bill statement in September (NT dollar)")
    BILL_AMT2: float = Field(default=18000.0, description="Amount of bill statement in August")
    BILL_AMT3: float = Field(default=15000.0, description="Amount of bill statement in July")
    BILL_AMT4: float = Field(default=12000.0, description="Amount of bill statement in June")
    BILL_AMT5: float = Field(default=10000.0, description="Amount of bill statement in May")
    BILL_AMT6: float = Field(default=8000.0, description="Amount of bill statement in April")
    PAY_AMT1: float = Field(default=3000.0, ge=0, description="Amount paid in September (NT dollar)")
    PAY_AMT2: float = Field(default=2500.0, ge=0, description="Amount paid in August")
    PAY_AMT3: float = Field(default=2000.0, ge=0, description="Amount paid in July")
    PAY_AMT4: float = Field(default=1500.0, ge=0, description="Amount paid in June")
    PAY_AMT5: float = Field(default=1200.0, ge=0, description="Amount paid in May")
    PAY_AMT6: float = Field(default=1000.0, ge=0, description="Amount paid in April")

class PredictionResponse(BaseModel):
    """Production response schema."""
    dataset: str
    model_name: str
    default_probability: float
    decision: int = Field(description="0 = Approved / Low Risk, 1 = High Risk / Default Warning")
    decision_label: str
    risk_level: str
    threshold_applied: float
    optimization_objective: str
