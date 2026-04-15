from pydantic import BaseModel, Field
from typing import Dict, Literal


CountryType = Literal["USA", "UK", "France", "Russia", "China", "India", "Germany", "Brazil", "Japan", "Australia"]
IndustryType = Literal["Banking", "Healthcare", "IT", "Government", "Defense", "Retail", "Telecom", "Education", "Energy", "Media"]
AttackSourceType = Literal["Hacker Group", "Nation-State", "Insider Threat", "Unknown", "Lone Wolf"]
VulnerabilityType = Literal["Weak Passwords", "Zero-Day Exploit", "Unpatched Software", "Social Engineering", "Cloud Misconfiguration"]
DefenseType = Literal["Firewall", "AI-Based Detection", "Encryption", "Multi-Factor Authentication", "VPN"]
AttackTypeType = Literal["Ransomware", "Phishing", "DDoS", "Malware", "SQL Injection", "Man-in-the-Middle"]


class ClassificationInput(BaseModel):
    Country: CountryType = Field(..., example="France")
    Year: int = Field(..., ge=2015, le=2030, example=2023)
    Target_Industry: IndustryType = Field(..., example="Banking")
    Financial_Loss: float = Field(..., ge=0, example=45.5)
    Number_of_Affected_Users: int = Field(..., ge=0, example=500000)
    Attack_Source: AttackSourceType = Field(..., example="Hacker Group")
    Security_Vulnerability_Type: VulnerabilityType = Field(..., example="Weak Passwords")
    Defense_Mechanism_Used: DefenseType = Field(..., example="Firewall")
    Incident_Resolution_Time: float = Field(..., ge=0, example=30.0)


class RegressionInput(BaseModel):
    Country: CountryType = Field(..., example="France")
    Year: int = Field(..., ge=2015, le=2030, example=2023)
    Attack_Type: AttackTypeType = Field(..., example="Ransomware")
    Target_Industry: IndustryType = Field(..., example="Banking")
    Number_of_Affected_Users: int = Field(..., ge=0, example=500000)
    Attack_Source: AttackSourceType = Field(..., example="Hacker Group")
    Security_Vulnerability_Type: VulnerabilityType = Field(..., example="Weak Passwords")
    Defense_Mechanism_Used: DefenseType = Field(..., example="Firewall")
    Incident_Resolution_Time: float = Field(..., ge=0, example=30.0)


class ClassificationOutput(BaseModel):
    prediction: str
    confidence: float
    class_probabilities: Dict[str, float]


class RegressionOutput(BaseModel):
    predicted_financial_loss: float
    unit: str = "Million $"


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
