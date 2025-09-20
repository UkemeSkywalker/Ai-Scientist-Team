import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar, List
from pydantic import BaseModel, ValidationError
import logging

from ..models.workflow import WorkflowState, AgentType, ResearchContext
from ..models.research import ResearchFindings
from ..models.data import DataContext
from ..models.experiment import ExperimentResults
from ..models.critic import CriticalEvaluation
from ..models.visualization import VisualizationResults

T = TypeVar('T', bound=BaseModel)

logger = logging.getLogger(__name__)


class SharedMemoryError(Exception):
    """Base exception for shared memory operations."""
    pass


class ValidationError(SharedMemoryError):
    """Raised when data validation fails."""
    pass


class VersionMismatchError(SharedMemoryError):
    """Raised when version conflicts occur."""
    pass


class SharedMemory:
    """Local file-based shared memory system for agent communication with validation and versioning."""
    
    def __init__(self, storage_dir: str = "data/shared_memory"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._version = 1
        
        # Type mapping for automatic model validation
        self._type_mapping = {
            'ResearchFindings': ResearchFindings,
            'DataContext': DataContext,
            'ExperimentResults': ExperimentResults,
            'CriticalEvaluation': CriticalEvaluation,
            'VisualizationResults': VisualizationResults,
            'ResearchContext': ResearchContext,
            'WorkflowState': WorkflowState
        }
    
    def _get_session_dir(self, session_id: str) -> Path:
        """Get the directory for a specific session."""
        if not session_id or not session_id.strip():
            raise SharedMemoryError("Session ID cannot be empty")
        
        session_dir = self.storage_dir / session_id
        session_dir.mkdir(exist_ok=True)
        return session_dir
    
    def _get_file_path(self, session_id: str, key: str) -> Path:
        """Get the file path for a specific key in a session."""
        if not key or not key.strip():
            raise SharedMemoryError("Key cannot be empty")
        
        return self._get_session_dir(session_id) / f"{key}.json"
    
    def _get_backup_path(self, session_id: str, key: str, version: int) -> Path:
        """Get backup file path for versioning."""
        session_dir = self._get_session_dir(session_id)
        backup_dir = session_dir / "backups"
        backup_dir.mkdir(exist_ok=True)
        return backup_dir / f"{key}_v{version}.json"
    
    def _validate_data(self, data: Any, expected_type: Optional[str] = None) -> None:
        """Validate data before writing."""
        if data is None:
            raise ValidationError("Data cannot be None")
        
        if isinstance(data, BaseModel):
            try:
                # Validate the model
                data.model_validate(data.model_dump())
            except Exception as e:
                raise ValidationError(f"Model validation failed: {e}")
        
        if expected_type and hasattr(data, '__class__'):
            if data.__class__.__name__ != expected_type:
                logger.warning(f"Type mismatch: expected {expected_type}, got {data.__class__.__name__}")
    
    def _create_backup(self, session_id: str, key: str) -> None:
        """Create a backup of existing data before overwriting."""
        file_path = self._get_file_path(session_id, key)
        if not file_path.exists():
            return
        
        try:
            with open(file_path, 'r') as f:
                existing_payload = json.load(f)
            
            version = existing_payload.get("version", 1)
            backup_path = self._get_backup_path(session_id, key, version)
            
            with open(backup_path, 'w') as f:
                json.dump(existing_payload, f, indent=2, default=str)
            
            logger.debug(f"Created backup: {backup_path}")
        except Exception as e:
            logger.warning(f"Failed to create backup for {key}: {e}")
    
    def write(self, session_id: str, key: str, data: Any, create_backup: bool = True) -> None:
        """Write data to shared memory with validation and versioning."""
        try:
            # Validate inputs
            self._validate_data(data)
            
            # Create backup if requested
            if create_backup:
                self._create_backup(session_id, key)
            
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
                "type": type(data).__name__,
                "version": self._version,
                "key": key,
                "session_id": session_id
            }
            
            # Write atomically by writing to temp file first
            temp_path = file_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(payload, f, indent=2, default=str)
            
            # Move temp file to final location
            temp_path.replace(file_path)
            
            logger.debug(f"Successfully wrote {key} to session {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to write {key} to session {session_id}: {e}")
            raise SharedMemoryError(f"Write operation failed: {e}")
    
    def read(self, session_id: str, key: str, model_class: Optional[Type[T]] = None, 
             validate: bool = True) -> Optional[T]:
        """Read data from shared memory with optional validation."""
        file_path = self._get_file_path(session_id, key)
        
        if not file_path.exists():
            logger.debug(f"File not found: {file_path}")
            return None
        
        try:
            with open(file_path, 'r') as f:
                payload = json.load(f)
            
            data = payload.get("data")
            if data is None:
                logger.warning(f"No data found in {key}")
                return None
            
            # Auto-detect model class from type mapping
            if not model_class and validate:
                data_type = payload.get("type")
                if data_type in self._type_mapping:
                    model_class = self._type_mapping[data_type]
            
            # Convert back to Pydantic model if requested
            if model_class and issubclass(model_class, BaseModel):
                try:
                    return model_class.model_validate(data)
                except Exception as e:
                    if validate:
                        raise ValidationError(f"Failed to validate {key} as {model_class.__name__}: {e}")
                    logger.warning(f"Validation failed for {key}, returning raw data: {e}")
                    return data
            
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error reading {key}: {e}")
            raise SharedMemoryError(f"Corrupted data in {key}: {e}")
        except Exception as e:
            logger.error(f"Error reading {key}: {e}")
            raise SharedMemoryError(f"Read operation failed: {e}")
    
    def create_research_context(self, session_id: str, query: str) -> ResearchContext:
        """Create a new research context for a session."""
        context = ResearchContext(
            session_id=session_id,
            query=query,
            timestamp=datetime.now()
        )
        self.write(session_id, "research_context", context)
        return context
    
    def get_research_context(self, session_id: str) -> Optional[ResearchContext]:
        """Get the research context for a session."""
        return self.read(session_id, "research_context", ResearchContext)
    
    def update_research_context(self, context: ResearchContext) -> None:
        """Update the research context with new data."""
        context.updated_at = datetime.now()
        context.version += 1
        self.write(context.session_id, "research_context", context)
    
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
    
    def write_agent_results(self, session_id: str, agent_type: AgentType, results: Any) -> None:
        """Write agent results to both workflow state and research context."""
        # Update workflow state
        self.update_agent_status(session_id, agent_type, "completed", 100, results)
        
        # Update research context
        context = self.get_research_context(session_id)
        if context:
            if agent_type == AgentType.RESEARCH:
                context.research_findings = results
            elif agent_type == AgentType.DATA:
                context.data_context = results
            elif agent_type == AgentType.EXPERIMENT:
                context.experiment_results = results
            elif agent_type == AgentType.CRITIC:
                context.critical_evaluation = results
            elif agent_type == AgentType.VISUALIZATION:
                context.visualization_results = results
            
            self.update_research_context(context)
    
    def get_context(self, session_id: str) -> Dict[str, Any]:
        """Get all context data for a session."""
        session_dir = self._get_session_dir(session_id)
        context = {}
        
        for file_path in session_dir.glob("*.json"):
            if file_path.name.startswith('.'):  # Skip hidden files
                continue
                
            key = file_path.stem
            try:
                with open(file_path, 'r') as f:
                    payload = json.load(f)
                context[key] = {
                    'data': payload.get("data"),
                    'timestamp': payload.get("timestamp"),
                    'type': payload.get("type"),
                    'version': payload.get("version", 1)
                }
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"Failed to read {file_path}: {e}")
                continue
        
        return context
    
    def get_session_metadata(self, session_id: str) -> Dict[str, Any]:
        """Get metadata about a session."""
        session_dir = self._get_session_dir(session_id)
        if not session_dir.exists():
            return {}
        
        files = list(session_dir.glob("*.json"))
        backup_files = list((session_dir / "backups").glob("*.json")) if (session_dir / "backups").exists() else []
        
        return {
            'session_id': session_id,
            'file_count': len(files),
            'backup_count': len(backup_files),
            'total_size': sum(f.stat().st_size for f in files),
            'created_at': datetime.fromtimestamp(session_dir.stat().st_ctime).isoformat(),
            'modified_at': datetime.fromtimestamp(session_dir.stat().st_mtime).isoformat(),
            'files': [f.stem for f in files]
        }
    
    def clear_session(self, session_id: str, keep_backups: bool = False) -> None:
        """Clear all data for a session."""
        session_dir = self._get_session_dir(session_id)
        
        if not keep_backups:
            # Remove entire session directory
            if session_dir.exists():
                shutil.rmtree(session_dir)
                logger.info(f"Cleared session {session_id} completely")
        else:
            # Remove only main files, keep backups
            for file_path in session_dir.glob("*.json"):
                file_path.unlink()
            logger.info(f"Cleared session {session_id} (kept backups)")
    
    def list_sessions(self) -> List[str]:
        """List all active sessions."""
        return [d.name for d in self.storage_dir.iterdir() if d.is_dir()]
    
    def get_version_history(self, session_id: str, key: str) -> List[Dict[str, Any]]:
        """Get version history for a specific key."""
        session_dir = self._get_session_dir(session_id)
        backup_dir = session_dir / "backups"
        
        if not backup_dir.exists():
            return []
        
        versions = []
        for backup_file in backup_dir.glob(f"{key}_v*.json"):
            try:
                with open(backup_file, 'r') as f:
                    payload = json.load(f)
                versions.append({
                    'version': payload.get('version', 1),
                    'timestamp': payload.get('timestamp'),
                    'file_path': str(backup_file),
                    'type': payload.get('type')
                })
            except Exception as e:
                logger.warning(f"Failed to read backup {backup_file}: {e}")
        
        return sorted(versions, key=lambda x: x['version'])
    
    def restore_version(self, session_id: str, key: str, version: int) -> bool:
        """Restore a specific version of data."""
        backup_path = self._get_backup_path(session_id, key, version)
        
        if not backup_path.exists():
            logger.error(f"Backup version {version} not found for {key}")
            return False
        
        try:
            # Create backup of current version
            self._create_backup(session_id, key)
            
            # Restore from backup
            file_path = self._get_file_path(session_id, key)
            shutil.copy2(backup_path, file_path)
            
            logger.info(f"Restored {key} to version {version}")
            return True
        except Exception as e:
            logger.error(f"Failed to restore {key} to version {version}: {e}")
            return False
    
    def validate_session(self, session_id: str) -> Dict[str, Any]:
        """Validate all data in a session."""
        validation_results = {
            'session_id': session_id,
            'valid': True,
            'errors': [],
            'warnings': [],
            'file_results': {}
        }
        
        try:
            context = self.get_context(session_id)
            
            for key, file_data in context.items():
                file_result = {
                    'valid': True,
                    'errors': [],
                    'warnings': []
                }
                
                try:
                    data_type = file_data.get('type')
                    if data_type in self._type_mapping:
                        model_class = self._type_mapping[data_type]
                        model_class.model_validate(file_data['data'])
                    else:
                        file_result['warnings'].append(f"Unknown data type: {data_type}")
                        
                except Exception as e:
                    file_result['valid'] = False
                    file_result['errors'].append(str(e))
                    validation_results['valid'] = False
                
                validation_results['file_results'][key] = file_result
                
        except Exception as e:
            validation_results['valid'] = False
            validation_results['errors'].append(f"Session validation failed: {e}")
        
        return validation_results
    
    def update_context(self, session_id: str, context_update: Dict[str, Any]) -> None:
        """Update context for a session with Strands integration"""
        for key, value in context_update.items():
            self.write(session_id, key, value)
        
        logger.debug(f"Updated context for session {session_id} with keys: {list(context_update.keys())}")
    
    def get_strands_context(self, session_id: str) -> Dict[str, Any]:
        """Get context formatted for Strands agent consumption"""
        context = self.get_context(session_id)
        
        # Extract just the data portion for Strands agents
        strands_context = {}
        for key, file_data in context.items():
            if isinstance(file_data, dict) and 'data' in file_data:
                strands_context[key] = file_data['data']
            else:
                strands_context[key] = file_data
        
        return strands_context
    
    def store_strands_conversation(self, session_id: str, conversation_history: List[Dict[str, Any]]) -> None:
        """Store Strands agent conversation history"""
        self.write(session_id, "strands_conversation", {
            "messages": conversation_history,
            "message_count": len(conversation_history)
        })
    
    def get_strands_conversation(self, session_id: str) -> List[Dict[str, Any]]:
        """Get Strands agent conversation history"""
        conversation_data = self.read(session_id, "strands_conversation")
        if conversation_data and isinstance(conversation_data, dict):
            return conversation_data.get("messages", [])
        return []