from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class Hypothesis(BaseModel):
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    testable: bool = True
    variables: List[str] = Field(default_factory=list)
    expected_outcome: Optional[str] = None


class LiteratureSource(BaseModel):
    title: str
    authors: List[str]
    publication_date: Optional[datetime] = None
    source: str  # arxiv, pubmed, etc.
    url: Optional[str] = None
    relevance_score: float = Field(ge=0.0, le=1.0)
    abstract: Optional[str] = None


class ResearchFindings(BaseModel):
    hypotheses: List[Hypothesis]
    literature_sources: List[LiteratureSource] = Field(default_factory=list)
    research_gaps: List[str] = Field(default_factory=list)
    key_concepts: List[str] = Field(default_factory=list)
    confidence_score: float = Field(ge=0.0, le=1.0)
    methodology_suggestions: List[str] = Field(default_factory=list)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }