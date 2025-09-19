import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar
from pydantic import BaseModel

from ..models.workflow import WorkflowState, AgentType

T = TypeVar('T', bound=BaseModel)


class SharedMemory:
    """Local file-based shared memory system for agent communication."""
    
    def __init__(self, storage_dir: str = "data/shared_memory"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_session_dir(self, session_id: str) -> Path:
        """Get the directory for a specific session."""
        session_dir = self.storage_dir / session_id
        session_dir.mkdir(exist_ok=True)
        return session_dir
    
    def _get_file_path(self, session_id: str, key: str) -> Path:
        """Get the file path for a specific key in a session."""
        return self._get_session_dir(session_id) / f"{key}.json"
    
    def write(self, session_id: str, key: str, data: Any) -> None:
        """Write data to shared memory."""
        file_path = self._get_file_path(session_id, key)
        
        # Convert Pydantic models to dict
        if isinstance(data, BaseModel):
            data_dict = data.model_dump()
        else:
            data_dict = data
        
        # Add metadata
        payload = {
            "data": data_dict,
            "timestamp": datetime.now().isoformat(),
            "type": type(data).__name__
        }
        
        with open(file_path, 'w') as f:
            json.dump(payload, f, indent=2, default=str)
    
    def read(self, session_id: str, key: str, model_class: Optional[Type[T]] = None) -> Optional[T]:
        """Read data from shared memory."""
        file_path = self._get_file_path(session_id, key)
        
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'r') as f:
                payload = json.load(f)
            
            data = payload.get("data")
            if data is None:
                return None
            
            # Convert back to Pydantic model if requested
            if model_class and issubclass(model_class, BaseModel):
                return model_class.model_validate(data)
            
            return data
        except (json.JSONDecodeError, Exception):
            return None
    
    def update_workflow_state(self, workflow_state: WorkflowState) -> None:
        """Update the workflow state in shared memory."""
        self.write(workflow_state.session_id, "workflow_state", workflow_state)
    
    def get_workflow_state(self, session_id: str) -> Optional[WorkflowState]:
        """Get the current workflow state."""
        return self.read(session_id, "workflow_state", WorkflowState)
    
    def update_agent_status(self, session_id: str, agent_type: AgentType, 
                           status: str, progress: int = 0, results: Any = None, 
                           error: Optional[str] = None) -> None:
        """Update the status of a specific agent."""
        workflow_state = self.get_workflow_state(session_id)
        if not workflow_state:
            return
        
        agent_status = workflow_state.agents[agent_type]
        agent_status.status = status
        agent_status.progress = progress
        if results is not None:
            agent_status.results = results
        if error:
            agent_status.error = error
        
        if status == "running" and not agent_status.start_time:
            agent_status.start_time = datetime.now()
        elif status in ["completed", "failed"]:
            agent_status.end_time = datetime.now()
        
        self.update_workflow_state(workflow_state)
    
    def get_context(self, session_id: str) -> Dict[str, Any]:
        """Get all context data for a session."""
        session_dir = self._get_session_dir(session_id)
        context = {}
        
        for file_path in session_dir.glob("*.json"):
            key = file_path.stem
            try:
                with open(file_path, 'r') as f:
                    payload = json.load(f)
                context[key] = payload.get("data")
            except (json.JSONDecodeError, Exception):
                continue
        
        return context
    
    def clear_session(self, session_id: str) -> None:
        """Clear all data for a session."""
        session_dir = self._get_session_dir(session_id)
        for file_path in session_dir.glob("*.json"):
            file_path.unlink()
        
        try:
            session_dir.rmdir()
        except OSError:
            pass  # Directory not empty or doesn't exist
    
    def list_sessions(self) -> list[str]:
        """List all active sessions."""
        return [d.name for d in self.storage_dir.iterdir() if d.is_dir()]