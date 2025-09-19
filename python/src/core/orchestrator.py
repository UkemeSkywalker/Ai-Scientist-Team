import asyncio
from datetime import datetime
from typing import Dict, List, Optional
from uuid import uuid4

from ..models.workflow import WorkflowState, AgentType, WorkflowStatusType, AgentStatusType
from .shared_memory import SharedMemory
from .logger import AgentLogger


class MockAgent:
    """Mock agent for testing workflow orchestration."""
    
    def __init__(self, agent_type: AgentType, shared_memory: SharedMemory):
        self.agent_type = agent_type
        self.shared_memory = shared_memory
        self.logger = None
    
    async def execute(self, session_id: str, query: str) -> Dict:
        """Execute the mock agent with simulated work."""
        self.logger = AgentLogger(self.agent_type.value, session_id)
        
        self.logger.info(f"Starting {self.agent_type.value} agent execution")
        
        # Simulate progress updates
        for progress in [20, 40, 60, 80, 100]:
            await asyncio.sleep(1)  # Simulate work
            self.shared_memory.update_agent_status(
                session_id, self.agent_type, AgentStatusType.RUNNING, progress
            )
            self.logger.info(f"Progress: {progress}%")
        
        # Generate mock results based on agent type
        results = self._generate_mock_results(query)
        
        self.shared_memory.update_agent_status(
            session_id, self.agent_type, AgentStatusType.COMPLETED, 100, results
        )
        
        self.logger.info(f"Completed {self.agent_type.value} agent execution")
        return results
    
    def _generate_mock_results(self, query: str) -> Dict:
        """Generate mock results based on agent type."""
        if self.agent_type == AgentType.RESEARCH:
            return {
                "hypotheses": [
                    f"Hypothesis 1 related to: {query[:50]}...",
                    f"Hypothesis 2 exploring: {query[:50]}..."
                ],
                "literature_count": 42,
                "key_findings": f"Found relevant research on {query[:30]}..."
            }
        elif self.agent_type == AgentType.DATA:
            return {
                "datasets_found": 3,
                "total_samples": 125000,
                "quality_score": 0.87,
                "sources": ["Kaggle", "HuggingFace", "AWS Open Data"]
            }
        elif self.agent_type == AgentType.EXPERIMENT:
            return {
                "experiments_run": 5,
                "best_accuracy": 0.94,
                "statistical_significance": 0.001,
                "model_type": "Random Forest Classifier"
            }
        elif self.agent_type == AgentType.CRITIC:
            return {
                "overall_confidence": 0.82,
                "limitations": ["Limited demographic diversity", "Potential selection bias"],
                "recommendations": ["Expand dataset diversity", "Cross-validate with external data"]
            }
        elif self.agent_type == AgentType.VISUALIZATION:
            return {
                "charts_generated": 8,
                "dashboard_url": "/mock-dashboard",
                "report_generated": True,
                "export_formats": ["PDF", "HTML", "Interactive"]
            }
        
        return {}


class WorkflowOrchestrator:
    """Orchestrates the execution of multiple agents in sequence."""
    
    def __init__(self, shared_memory: Optional[SharedMemory] = None):
        self.shared_memory = shared_memory or SharedMemory()
        self.agents: Dict[AgentType, MockAgent] = {
            agent_type: MockAgent(agent_type, self.shared_memory)
            for agent_type in AgentType
        }
        self.logger = None
    
    async def execute_workflow(self, query: str, session_id: Optional[str] = None) -> WorkflowState:
        """Execute the complete research workflow."""
        if not session_id:
            session_id = f"session_{uuid4().hex[:8]}"
        
        self.logger = AgentLogger("orchestrator", session_id)
        
        # Initialize workflow state
        workflow_state = WorkflowState(
            session_id=session_id,
            query=query,
            status=WorkflowStatusType.RUNNING,
            start_time=datetime.now()
        )
        
        self.shared_memory.update_workflow_state(workflow_state)
        self.logger.info("Workflow started", query=query)
        
        try:
            # Execute agents in sequence
            agent_sequence = [
                AgentType.RESEARCH,
                AgentType.DATA,
                AgentType.EXPERIMENT,
                AgentType.CRITIC,
                AgentType.VISUALIZATION
            ]
            
            for agent_type in agent_sequence:
                workflow_state.current_agent = agent_type
                self.shared_memory.update_workflow_state(workflow_state)
                
                # Update agent status to running
                self.shared_memory.update_agent_status(
                    session_id, agent_type, AgentStatusType.RUNNING
                )
                
                # Execute agent
                agent = self.agents[agent_type]
                try:
                    await agent.execute(session_id, query)
                except Exception as e:
                    self.logger.error(f"Agent {agent_type.value} failed", error=str(e))
                    self.shared_memory.update_agent_status(
                        session_id, agent_type, AgentStatusType.FAILED, error=str(e)
                    )
                    raise
            
            # Mark workflow as completed
            workflow_state = self.shared_memory.get_workflow_state(session_id)
            if workflow_state:
                workflow_state.status = WorkflowStatusType.COMPLETED
                workflow_state.end_time = datetime.now()
                self.shared_memory.update_workflow_state(workflow_state)
            
            self.logger.info("Workflow completed successfully")
            return workflow_state
            
        except Exception as e:
            # Mark workflow as failed
            workflow_state = self.shared_memory.get_workflow_state(session_id)
            if workflow_state:
                workflow_state.status = WorkflowStatusType.FAILED
                workflow_state.error = str(e)
                workflow_state.end_time = datetime.now()
                self.shared_memory.update_workflow_state(workflow_state)
            
            self.logger.error("Workflow failed", error=str(e))
            raise
    
    def get_workflow_status(self, session_id: str) -> Optional[WorkflowState]:
        """Get the current status of a workflow."""
        return self.shared_memory.get_workflow_state(session_id)
    
    def list_active_workflows(self) -> List[str]:
        """List all active workflow sessions."""
        return self.shared_memory.list_sessions()