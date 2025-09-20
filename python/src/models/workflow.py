from datetime import datetime
from enum import Enum
from typing import Dict, Optional, Any, List
from pydantic import BaseModel, Field


class AgentType(str, Enum):
    RESEARCH = "research"
    DATA = "data"
    EXPERIMENT = "experiment"
    CRITIC = "critic"
    VISUALIZATION = "visualization"


class AgentStatusType(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class WorkflowStatusType(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentStatus(BaseModel):
    status: AgentStatusType = AgentStatusType.PENDING
    progress: int = Field(default=0, ge=0, le=100)
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


class ResearchContext(BaseModel):
    """Comprehensive research context that contains all agent outputs."""
    session_id: str
    query: str
    timestamp: datetime = Field(default_factory=datetime.now)
    status: WorkflowStatusType = WorkflowStatusType.IDLE
    current_agent: Optional[AgentType] = None
    version: int = 1
    
    # Agent outputs - imported from respective models
    research_findings: Optional[Any] = None  # ResearchFindings
    data_context: Optional[Any] = None       # DataContext  
    experiment_results: Optional[Any] = None # ExperimentResults
    critical_evaluation: Optional[Any] = None # CriticalEvaluation
    visualization_results: Optional[Any] = None # VisualizationResults
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class WorkflowState(BaseModel):
    session_id: str
    query: str
    status: WorkflowStatusType = WorkflowStatusType.IDLE
    current_agent: Optional[AgentType] = None
    agents: Dict[AgentType, AgentStatus] = Field(
        default_factory=lambda: {
            AgentType.RESEARCH: AgentStatus(),
            AgentType.DATA: AgentStatus(),
            AgentType.EXPERIMENT: AgentStatus(),
            AgentType.CRITIC: AgentStatus(),
            AgentType.VISUALIZATION: AgentStatus(),
        }
    )
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    estimated_duration: int = 300  # seconds
    error: Optional[str] = None

    class Config:
        use_enum_values = True