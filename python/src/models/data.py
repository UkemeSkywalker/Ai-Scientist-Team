from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class DatasetMetadata(BaseModel):
    name: str
    source: str  # kaggle, huggingface, aws_open_data
    url: Optional[str] = None
    description: Optional[str] = None
    size_bytes: Optional[int] = None
    num_samples: Optional[int] = None
    num_features: Optional[int] = None
    file_format: Optional[str] = None
    license: Optional[str] = None
    last_updated: Optional[datetime] = None
    relevance_score: float = Field(ge=0.0, le=1.0)


class DataQualityMetrics(BaseModel):
    completeness: float = Field(ge=0.0, le=1.0)
    consistency: float = Field(ge=0.0, le=1.0)
    accuracy: float = Field(ge=0.0, le=1.0)
    validity: float = Field(ge=0.0, le=1.0)
    overall_score: float = Field(ge=0.0, le=1.0)
    issues_found: List[str] = Field(default_factory=list)


class PreprocessingStep(BaseModel):
    step_name: str
    description: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    rows_before: Optional[int] = None
    rows_after: Optional[int] = None
    execution_time: Optional[float] = None


class S3Location(BaseModel):
    bucket: str
    key: str
    region: str = "us-east-1"
    size_bytes: Optional[int] = None
    last_modified: Optional[datetime] = None
    metadata: Dict[str, str] = Field(default_factory=dict)


class DataContext(BaseModel):
    datasets: List[DatasetMetadata]
    quality_metrics: Optional[DataQualityMetrics] = None
    preprocessing_steps: List[PreprocessingStep] = Field(default_factory=list)
    s3_locations: List[S3Location] = Field(default_factory=list)
    total_samples: int = 0
    total_features: int = 0
    data_types: Dict[str, str] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }