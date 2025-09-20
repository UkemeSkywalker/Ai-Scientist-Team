"""
Strands SDK-based orchestrator for the AI Scientist Team multi-agent system
"""

import asyncio
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4
import structlog

try:
    from strands import Agent
    STRANDS_AVAILABLE = True
except ImportError:
    STRANDS_AVAILABLE = False
    Agent = None

from ..models.workflow import WorkflowState, AgentType, WorkflowStatusType, AgentStatusType
from ..core.shared_memory import SharedMemory
from ..core.logger import get_logger, AgentLogger
from ..integrations.strands_config import strands_config
from ..tools.mock_tools import (
    mock_research_tool,
    mock_data_collection_tool,
    mock_experiment_tool,
    mock_critic_tool,
    mock_visualization_tool
)

logger = get_logger(__name__)

@dataclass
class AgentResult:
    """Result from an agent execution"""
    agent_name: str
    success: bool
    output: Dict[str, Any]
    error: Optional[str] = None

class StrandsOrchestrator:
    """
    Strands SDK-based orchestrator for coordinating AI agents in the research workflow
    """
    
    def __init__(self, shared_memory: Optional[SharedMemory] = None):
        self.shared_memory = shared_memory or SharedMemory()
        self.workflow_state = None
        self.orchestrator_agent = None
        self.logger = None
        self._initialize_orchestrator()
        
    def _initialize_orchestrator(self):
        """Initialize the Strands orchestrator agent with mock tools"""
        if not STRANDS_AVAILABLE:
            logger.warning("Strands SDK not available, using mock mode")
            return
            
        try:
            # Validate configuration
            if not strands_config.validate_config():
                logger.error("Strands configuration validation failed")
                return
            
            # Create orchestrator agent with mock tools as Strands tools
            self.orchestrator_agent = Agent(
                model=strands_config.model_id,
                system_prompt="""You are the AI Scientist Team orchestrator. You coordinate specialized research agents to conduct comprehensive research analysis.

Your role is to:
1. Route research queries to appropriate specialized agents
2. Manage the workflow sequence: Research → Data → Experiment → Critic → Visualization
3. Ensure each agent builds upon previous agents' work
4. Maintain context and state throughout the workflow

Available tools:
- mock_research_tool: For literature search and hypothesis generation
- mock_data_collection_tool: For dataset discovery and processing
- mock_experiment_tool: For running experiments and analysis
- mock_critic_tool: For critical evaluation of results
- mock_visualization_tool: For creating charts and reports

Always use the tools in sequence and pass relevant context between them.""",
                tools=[
                    self._create_strands_tool("mock_research_tool", mock_research_tool),
                    self._create_strands_tool("mock_data_collection_tool", mock_data_collection_tool),
                    self._create_strands_tool("mock_experiment_tool", mock_experiment_tool),
                    self._create_strands_tool("mock_critic_tool", mock_critic_tool),
                    self._create_strands_tool("mock_visualization_tool", mock_visualization_tool)
                ]
            )
            
            logger.info("Strands orchestrator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Strands orchestrator: {str(e)}")
            self.orchestrator_agent = None
    
    def _create_strands_tool(self, name: str, func):
        """Create a Strands tool from a Python function"""
        # For now, return the function directly
        # In a real implementation, this would use @strands.tool decorator
        return func
        
    async def execute_workflow(self, query: str, session_id: Optional[str] = None) -> WorkflowState:
        """
        Execute the complete research workflow using Strands orchestrator
        
        Args:
            query: Research query to process
            session_id: Unique session identifier
            
        Returns:
            WorkflowState with complete results
        """
        if not session_id:
            session_id = f"session_{uuid4().hex[:8]}"
            
        self.logger = AgentLogger("strands_orchestrator", session_id)
        
        logger.info(f"Starting Strands workflow execution for query: {query}")
        
        # Initialize workflow state
        workflow_state = WorkflowState(
            session_id=session_id,
            query=query,
            status=WorkflowStatusType.RUNNING,
            start_time=datetime.now()
        )
        
        self.shared_memory.update_workflow_state(workflow_state)
        self.logger.info("Strands workflow started", query=query)
        
        try:
            if self.orchestrator_agent and STRANDS_AVAILABLE:
                # Use Strands agent for orchestration
                results = await self._execute_strands_workflow(query, session_id)
            else:
                # Fallback to mock execution
                results = await self._execute_mock_workflow(query, session_id)
            
            # Update workflow state with results
            workflow_state = self.shared_memory.get_workflow_state(session_id)
            if workflow_state:
                workflow_state.status = WorkflowStatusType.COMPLETED
                workflow_state.end_time = datetime.now()
                self.shared_memory.update_workflow_state(workflow_state)
            
            self.logger.info("Strands workflow completed successfully")
            return workflow_state
            
        except Exception as e:
            # Mark workflow as failed
            workflow_state = self.shared_memory.get_workflow_state(session_id)
            if workflow_state:
                workflow_state.status = WorkflowStatusType.FAILED
                workflow_state.error = str(e)
                workflow_state.end_time = datetime.now()
                self.shared_memory.update_workflow_state(workflow_state)
            
            self.logger.error("Strands workflow failed", error=str(e))
            raise
    
    async def _execute_strands_workflow(self, query: str, session_id: str) -> Dict[str, Any]:
        """Execute workflow using Strands agent"""
        logger.info("Executing workflow with Strands agent")
        
        # Create comprehensive prompt for the orchestrator
        workflow_prompt = f"""
        Execute a complete research workflow for the query: "{query}"
        
        Please follow this sequence:
        1. Use mock_research_tool to conduct literature research and generate hypotheses
        2. Use mock_data_collection_tool to find and process relevant datasets
        3. Use mock_experiment_tool to run experiments based on the research and data
        4. Use mock_critic_tool to critically evaluate the experimental results
        5. Use mock_visualization_tool to create visualizations and reports
        
        Pass relevant context between each step and provide a comprehensive summary at the end.
        """
        
        try:
            # Execute the workflow through Strands agent
            result = self.orchestrator_agent(workflow_prompt)
            
            # Parse the result and extract individual agent outputs
            workflow_results = self._parse_strands_result(str(result), session_id)
            
            # Update agent statuses in shared memory
            await self._update_agent_statuses(session_id, workflow_results)
            
            return workflow_results
            
        except Exception as e:
            logger.error(f"Strands workflow execution failed: {str(e)}")
            # Fallback to mock execution
            return await self._execute_mock_workflow(query, session_id)
    
    def _parse_strands_result(self, result_text: str, session_id: str) -> Dict[str, Any]:
        """Parse Strands agent result into structured workflow results"""
        # This is a simplified parser - in a real implementation,
        # you would have more sophisticated parsing logic
        
        results = {
            "orchestrator_output": result_text,
            "research": {"status": "completed", "source": "strands_agent", "output": result_text[:200]},
            "data": {"status": "completed", "source": "strands_agent", "output": result_text[:200]},
            "experiment": {"status": "completed", "source": "strands_agent", "output": result_text[:200]},
            "critic": {"status": "completed", "source": "strands_agent", "output": result_text[:200]},
            "visualization": {"status": "completed", "source": "strands_agent", "output": result_text[:200]}
        }
        
        return results
    
    async def _update_agent_statuses(self, session_id: str, results: Dict[str, Any]):
        """Update agent statuses in shared memory based on results"""
        agent_types = [
            AgentType.RESEARCH,
            AgentType.DATA,
            AgentType.EXPERIMENT,
            AgentType.CRITIC,
            AgentType.VISUALIZATION
        ]
        
        for agent_type in agent_types:
            agent_key = agent_type.value.lower()
            if agent_key in results:
                self.shared_memory.update_agent_status(
                    session_id, 
                    agent_type, 
                    AgentStatusType.COMPLETED, 
                    100,
                    results[agent_key]
                )
    
    async def _execute_mock_workflow(self, query: str, session_id: str) -> Dict[str, Any]:
        """Execute workflow using mock agents (fallback)"""
        logger.info("Executing workflow with mock agents")
        
        results = {}
        
        # Define agent sequence
        agent_sequence = [
            (AgentType.RESEARCH, "research", mock_research_tool),
            (AgentType.DATA, "data", mock_data_collection_tool),
            (AgentType.EXPERIMENT, "experiment", mock_experiment_tool),
            (AgentType.CRITIC, "critic", mock_critic_tool),
            (AgentType.VISUALIZATION, "visualization", mock_visualization_tool)
        ]
        
        for agent_type, agent_key, mock_tool in agent_sequence:
            # Update workflow state
            workflow_state = self.shared_memory.get_workflow_state(session_id)
            if workflow_state:
                workflow_state.current_agent = agent_type
                self.shared_memory.update_workflow_state(workflow_state)
            
            # Update agent status to running
            self.shared_memory.update_agent_status(
                session_id, agent_type, AgentStatusType.RUNNING
            )
            
            # Execute mock agent
            agent_result = await self._execute_mock_agent(agent_key, query, session_id, mock_tool)
            results[agent_key] = agent_result.output
            
            # Update agent status
            if agent_result.success:
                self.shared_memory.update_agent_status(
                    session_id, agent_type, AgentStatusType.COMPLETED, 100, agent_result.output
                )
            else:
                self.shared_memory.update_agent_status(
                    session_id, agent_type, AgentStatusType.FAILED, error=agent_result.error
                )
                raise Exception(f"Agent {agent_key} failed: {agent_result.error}")
        
        return results
    
    async def _execute_mock_agent(self, agent_name: str, query: str, session_id: str, mock_tool) -> AgentResult:
        """Execute a mock agent"""
        logger.info(f"Executing mock agent: {agent_name}")
        
        try:
            # Get current context
            context = self.shared_memory.get_context(session_id) or {}
            
            # Execute appropriate mock tool
            if agent_name == "research":
                output = mock_tool(query)
            elif agent_name == "data":
                output = mock_tool(query)
            elif agent_name == "experiment":
                output = mock_tool(query, json.dumps(context.get("data_result", {})))
            elif agent_name == "critic":
                output = mock_tool(json.dumps(context.get("experiment_result", {})))
            elif agent_name == "visualization":
                output = mock_tool(json.dumps(context))
            else:
                output = json.dumps({"status": "completed", "agent": agent_name})
            
            # Parse JSON output
            agent_output = json.loads(output) if isinstance(output, str) else output
            
            # Store agent result in shared memory
            self.shared_memory.update_context(session_id, {
                f"{agent_name}_result": agent_output
            })
            
            # Simulate progress updates
            for progress in [25, 50, 75, 100]:
                await asyncio.sleep(0.5)  # Simulate work
                if progress < 100:
                    agent_type = getattr(AgentType, agent_name.upper())
                    self.shared_memory.update_agent_status(
                        session_id, agent_type, AgentStatusType.RUNNING, progress
                    )
            
            return AgentResult(
                agent_name=agent_name,
                success=True,
                output=agent_output
            )
            
        except Exception as e:
            logger.error(f"Mock agent {agent_name} execution failed: {str(e)}")
            return AgentResult(
                agent_name=agent_name,
                success=False,
                output={},
                error=str(e)
            )
    
    def get_workflow_status(self, session_id: str) -> Optional[WorkflowState]:
        """Get the current status of a workflow."""
        return self.shared_memory.get_workflow_state(session_id)
    
    def list_active_workflows(self) -> List[str]:
        """List all active workflow sessions."""
        return self.shared_memory.list_sessions()

# Maintain backward compatibility
WorkflowOrchestrator = StrandsOrchestrator
Orchestrator = StrandsOrchestrator