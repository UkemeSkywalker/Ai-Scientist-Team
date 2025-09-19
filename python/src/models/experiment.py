from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field


class ExperimentConfig(BaseModel):
    experiment_type: str  # statistical, ml_modeling, simulation
    parameters: Dict[str, Any] = Field(default_factory=dict)
    target_variable: Optional[str] = None
    feature_columns: List[str] = Field(default_factory=list)
    test_size: float = Field(default=0.2, ge=0.1, le=0.5)
    random_state: int = 42


class ModelMetrics(BaseModel):
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    auc_roc: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    r2_score: Optional[float] = None
    custom_metrics: Dict[str, float] = Field(default_factory=dict)


class StatisticalTest(BaseModel):
    test_name: str
    test_statistic: float
    p_value: float
    critical_value: Optional[float] = None
    confidence_level: float = 0.95
    significant: bool
    interpretation: str


class ExperimentResult(BaseModel):
    experiment_id: str
    experiment_type: str
    config: ExperimentConfig
    metrics: Optional[ModelMetrics] = None
    statistical_tests: List[StatisticalTest] = Field(default_factory=list)
    model_artifacts: Dict[str, str] = Field(default_factory=dict)  # S3 paths
    execution_time: Optional[float] = None
    status: str = "completed"
    error: Optional[str] = None


class ExperimentPlan(BaseModel):
    experiments: List[ExperimentConfig]
    hypotheses_to_test: List[str]
    success_criteria: List[str]
    estimated_duration: int = 300  # seconds


class ExperimentResults(BaseModel):
    plan: ExperimentPlan
    results: List[ExperimentResult]
    best_result: Optional[ExperimentResult] = None
    summary_metrics: Dict[str, Union[float, str]] = Field(default_factory=dict)
    conclusions: List[str] = Field(default_factory=list)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }