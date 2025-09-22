from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class DatasetMetadata(BaseModel):
    name: str
    source: str  # kaggle, huggingface, aws_open_data
    category: Optional[str] = None  # research category (machine-learning, nlp, etc.)
    category_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    url: Optional[str] = None
    description: Optional[str] = None
    size_bytes: Optional[int] = None
    num_samples: Optional[int] = None
    num_features: Optional[int] = None
    file_format: Optional[str] = None
    license: Optional[str] = None
    last_updated: Optional[datetime] = None
    relevance_score: float = Field(ge=0.0, le=1.0)
    research_query: Optional[str] = None  # original query that led to this dataset


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
    category_path: Optional[str] = None  # e.g., "datasets/machine-learning/"
    region: str = "us-east-1"
    size_bytes: Optional[int] = None
    last_modified: Optional[datetime] = None
    metadata: Dict[str, str] = Field(default_factory=dict)
    reusable: bool = True  # indicates if this dataset can be reused


class DataContext(BaseModel):
    datasets: List[DatasetMetadata]
    primary_category: Optional[str] = None
    category_confidence: Optional[float] = None
    existing_datasets_reused: int = 0
    new_datasets_added: int = 0
    quality_metrics: Optional[DataQualityMetrics] = None
    preprocessing_steps: List[PreprocessingStep] = Field(default_factory=list)
    s3_locations: List[S3Location] = Field(default_factory=list)
    total_samples: int = 0
    total_features: int = 0
    data_types: Dict[str, str] = Field(default_factory=dict)
    reusability_achieved: bool = False
    organization_benefits: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class DatasetRecommendation(BaseModel):
    """Model for dataset recommendations from smart discovery"""
    type: str  # "existing" or "new"
    priority: str  # "high", "medium", "low"
    reason: str
    dataset: Dict[str, Any]
    action: str  # "reuse", "download_and_store", etc.
    target_category: Optional[str] = None


class SmartDiscoveryResult(BaseModel):
    """Model for smart dataset discovery results"""
    query: str
    category: str
    category_confidence: float
    strategy: Dict[str, Any]
    recommendations: List[DatasetRecommendation]
    existing_summary: Dict[str, Any]
    new_search_summary: Dict[str, Any]
    next_steps: List[str]
    status: str