from datetime import datetime
from enum import Enum
from typing import Dict, Optional, Any
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