from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ValidationIssue(BaseModel):
    severity: str  # low, medium, high, critical
    category: str  # statistical, methodological, data_quality, bias
    description: str
    recommendation: str
    confidence: float = Field(ge=0.0, le=1.0)


class BiasAssessment(BaseModel):
    bias_type: str  # selection, confirmation, survivorship, etc.
    detected: bool
    confidence: float = Field(ge=0.0, le=1.0)
    description: str
    mitigation_suggestions: List[str] = Field(default_factory=list)


class ReproducibilityCheck(BaseModel):
    aspect: str  # data, methodology, results, environment
    reproducible: bool
    confidence: float = Field(ge=0.0, le=1.0)
    issues: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class ValidationReport(BaseModel):
    overall_validity: float = Field(ge=0.0, le=1.0)
    issues: List[ValidationIssue] = Field(default_factory=list)
    bias_assessments: List[BiasAssessment] = Field(default_factory=list)
    reproducibility_checks: List[ReproducibilityCheck] = Field(default_factory=list)
    statistical_power: Optional[float] = None
    effect_size_assessment: Optional[str] = None


class Limitation(BaseModel):
    category: str  # data, methodology, scope, generalizability
    description: str
    impact: str  # low, medium, high
    potential_solutions: List[str] = Field(default_factory=list)


class Recommendation(BaseModel):
    priority: str  # low, medium, high, critical
    category: str  # data_collection, methodology, analysis, validation
    description: str
    rationale: str
    estimated_effort: Optional[str] = None


class CriticalEvaluation(BaseModel):
    validation_report: ValidationReport
    limitations: List[Limitation] = Field(default_factory=list)
    recommendations: List[Recommendation] = Field(default_factory=list)
    overall_confidence: float = Field(ge=0.0, le=1.0)
    research_quality_score: float = Field(ge=0.0, le=1.0)
    summary: str
    next_steps: List[str] = Field(default_factory=list)