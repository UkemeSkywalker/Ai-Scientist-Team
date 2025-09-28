"""
Strands Experiment Agent for automated ML experimentation and statistical analysis.
Handles experiment design, SageMaker integration, and results interpretation.
"""

import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import structlog

try:
    from strands import Agent
    STRANDS_AVAILABLE = True
except ImportError:
    STRANDS_AVAILABLE = False
    Agent = None

from ..tools.experiment_tools import (
    experiment_design_tool,
    sagemaker_training_tool,
    statistical_analysis_tool,
    results_interpretation_tool
)
from ..core.shared_memory import SharedMemory
from ..integrations.strands_config import strands_config
from ..core.logger import get_logger

logger = get_logger(__name__)


# System prompt for the Experiment Agent
EXPERIMENT_AGENT_PROMPT = """You are an expert Experiment Agent specializing in automated machine learning and statistical analysis. Your role is to design, execute, and interpret experiments based on research hypotheses and available data.

## Your Capabilities:
1. **Experiment Design**: Create comprehensive experiment plans based on hypotheses and data characteristics
2. **ML Training**: Execute machine learning experiments using SageMaker and scikit-learn
3. **Statistical Analysis**: Perform rigorous statistical tests using scipy and statsmodels
4. **Results Interpretation**: Generate actionable insights and recommendations from experimental results

## Your Tools:
- `experiment_design_tool`: Design appropriate experiments based on hypotheses and data context
- `sagemaker_training_tool`: Execute ML training jobs on AWS SageMaker
- `statistical_analysis_tool`: Perform statistical analysis using scipy and statsmodels
- `results_interpretation_tool`: Interpret results and generate insights

## Your Workflow:
1. **Design Phase**: Analyze hypotheses and data to create experiment plans
2. **Execution Phase**: Run ML training and statistical tests
3. **Analysis Phase**: Interpret results and generate insights
4. **Reporting Phase**: Provide comprehensive findings with confidence scores

## Quality Standards:
- Always include statistical significance testing
- Provide confidence intervals and effect sizes
- Consider multiple evaluation metrics
- Address potential biases and limitations
- Suggest follow-up experiments when appropriate

## Communication Style:
- Be precise and scientific in your language
- Explain statistical concepts clearly
- Provide actionable recommendations
- Include confidence scores for all findings
- Highlight both strengths and limitations of results

Remember: Your goal is to conduct rigorous, reproducible experiments that provide reliable insights for research questions."""


class ExperimentAgent:
    """
    Strands-powered Experiment Agent that conducts ML experiments and statistical analysis
    """
    
    def __init__(self, shared_memory: Optional[SharedMemory] = None):
        self.shared_memory = shared_memory or SharedMemory()
        self.agent = None
        self._initialize_agent()
    
    def _initialize_agent(self):
        """Initialize the Strands Experiment Agent with experiment tools"""
        if not STRANDS_AVAILABLE:
            logger.warning("Strands SDK not available, Experiment Agent will use direct tool calls")
            return
        
        try:
            # Validate Strands configuration
            if not strands_config.validate_config():
                logger.error("Strands configuration validation failed")
                return
            
            # Import conversation manager for better token management
            from strands.agent.conversation_manager import SlidingWindowConversationManager
            
            # Create Experiment Agent with specialized system prompt and tools
            self.agent = Agent(
                model=strands_config.model_id,
                system_prompt=EXPERIMENT_AGENT_PROMPT,
                tools=[
                    experiment_design_tool,
                    sagemaker_training_tool,
                    statistical_analysis_tool,
                    results_interpretation_tool
                ],
                name="Experiment Agent",
                description="Specialized agent for ML experimentation and statistical analysis",
                agent_id="experiment_agent",
                conversation_manager=SlidingWindowConversationManager(window_size=20, should_truncate_results=True)
            )
            
            logger.info("Experiment Agent initialized successfully with Strands SDK")
            
        except Exception as e:
            logger.error(f"Failed to initialize Experiment Agent: {str(e)}")
            self.agent = None


    async def execute_experiments(self, hypotheses: str, data_context: str, session_id: str) -> Dict[str, Any]:
        """
        Execute comprehensive experiment workflow for the given hypotheses and data
        
        Args:
            hypotheses: JSON string containing research hypotheses
            data_context: JSON string containing data metadata
            session_id: Session identifier for context management
            
        Returns:
            Dictionary containing experiment results
        """
        logger.info("Starting experiment execution", session_id=session_id)
        
        try:
            if self.agent and STRANDS_AVAILABLE:
                try:
                    # Use Strands agent for experiments
                    result = await self._execute_strands_experiments(hypotheses, data_context, session_id)
                except Exception as strands_error:
                    logger.warning(f"Strands agent failed, falling back to direct tools: {str(strands_error)}")
                    # Fallback to direct tool execution if Strands fails
                    result = await self._execute_direct_experiments(hypotheses, data_context, session_id)
            else:
                # Fallback to direct tool execution
                result = await self._execute_direct_experiments(hypotheses, data_context, session_id)
            
            # Store results in shared memory
            self.shared_memory.update_context(session_id, {
                "experiment_result": result,
                "experiment_timestamp": datetime.now().isoformat()
            })
            
            logger.info("Experiment execution completed successfully", session_id=session_id)
            return result
            
        except Exception as e:
            error_msg = f"Experiment execution failed: {str(e)}"
            logger.error("Experiment execution failed", error=error_msg, session_id=session_id)
            
            error_result = {
                "hypotheses": hypotheses,
                "data_context": data_context,
                "error": error_msg,
                "status": "failed",
                "timestamp": datetime.now().isoformat()
            }
            
            # Store error in shared memory
            self.shared_memory.update_context(session_id, {
                "experiment_result": error_result,
                "experiment_error": error_msg
            })
            
            return error_result
    
    async def _execute_strands_experiments(self, hypotheses: str, data_context: str, session_id: str) -> Dict[str, Any]:
        """Execute experiments using Strands agent conversation"""
        logger.info("Executing experiments with Strands agent", session_id=session_id)
        
        try:
            # Create a comprehensive prompt for the Strands agent
            experiment_prompt = f"""
            Please conduct a comprehensive experiment workflow using your tools:

            Research Hypotheses:
            {hypotheses}

            Data Context:
            {data_context}

            Please execute the following steps:
            1. Use experiment_design_tool to design appropriate experiments
            2. Use sagemaker_training_tool to execute ML training
            3. Use statistical_analysis_tool to perform statistical analysis
            4. Use results_interpretation_tool to interpret the results

            Provide comprehensive results for each step.
            """
            
            # Execute the conversation with the Strands agent
            logger.info("Starting Strands agent conversation", session_id=session_id)
            response = self.agent.run(experiment_prompt)
            
            # Parse the response to extract structured results
            # For now, we'll fall back to direct execution since Strands conversation parsing is complex
            logger.warning("Strands conversation completed, falling back to direct tools for structured results")
            return await self._execute_direct_experiments(hypotheses, data_context, session_id)
            
        except Exception as e:
            logger.error(f"Strands experiment execution failed: {str(e)}")
            # Fallback to direct execution
            return await self._execute_direct_experiments(hypotheses, data_context, session_id)
    
    async def _execute_direct_experiments(self, hypotheses: str, data_context: str, session_id: str) -> Dict[str, Any]:
        """Execute experiments using direct tool calls (fallback)"""
        logger.info("Executing experiments with direct tool calls", session_id=session_id)
        
        try:
            # Step 1: Design experiments
            logger.info("Designing experiments", session_id=session_id)
            design_results = experiment_design_tool(hypotheses, data_context)
            
            # Step 2: Execute ML training
            logger.info("Executing ML training", session_id=session_id)
            training_results = sagemaker_training_tool(design_results)
            
            # Step 3: Perform statistical analysis
            logger.info("Performing statistical analysis", session_id=session_id)
            analysis_data = json.dumps({
                "config": json.loads(design_results),
                "training_results": json.loads(training_results)
            })
            analysis_results = statistical_analysis_tool(analysis_data)
            
            # Step 4: Interpret results
            logger.info("Interpreting results", session_id=session_id)
            interpretation_data = json.dumps({
                "training_results": json.loads(training_results),
                "statistical_analysis": json.loads(analysis_results)
            })
            interpretation_results = results_interpretation_tool(interpretation_data)
            
            # Compile final results
            final_results = {
                "experiment_design": json.loads(design_results),
                "ml_training": json.loads(training_results),
                "statistical_analysis": json.loads(analysis_results),
                "interpretation": json.loads(interpretation_results),
                "execution_method": "direct_tools",
                "session_id": session_id,
                "status": "completed",
                "timestamp": datetime.now().isoformat()
            }
            
            return final_results
            
        except Exception as e:
            logger.error(f"Direct experiment execution failed: {str(e)}")
            raise
    
    def get_experiment_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get the current experiment status for a session"""
        context = self.shared_memory.get_context(session_id)
        if context and "experiment_result" in context:
            return context["experiment_result"]
        return None


def create_experiment_agent(shared_memory: Optional[SharedMemory] = None) -> ExperimentAgent:
    """
    Create and configure the Strands Experiment Agent.
    
    Args:
        shared_memory: Optional shared memory instance for context management
        
    Returns:
        Configured ExperimentAgent instance
    """
    return ExperimentAgent(shared_memory)


# Example usage and testing functions
async def test_experiment_agent():
    """Test the experiment agent with sample data."""
    
    # Create agent
    agent = create_experiment_agent()
    
    # Sample hypotheses
    sample_hypotheses = json.dumps({
        "hypotheses": [
            "Feature A is significantly correlated with the target variable",
            "Model performance can achieve >80% accuracy on this dataset",
            "There are no significant biases in the data distribution"
        ]
    })
    
    # Sample data context
    sample_data_context = json.dumps({
        "datasets": [
            {
                "name": "classification_dataset",
                "type": "supervised",
                "columns": ["feature_a", "feature_b", "feature_c", "target"],
                "target": "target",
                "shape": [1000, 4],
                "data_quality": "good"
            }
        ]
    })
    
    # Run workflow
    results = await agent.execute_experiments(sample_hypotheses, sample_data_context, "test_session")
    
    return results


if __name__ == "__main__":
    import asyncio
    # Test the agent
    test_results = asyncio.run(test_experiment_agent())
    print("Experiment Agent Test Results:")
    print(json.dumps(test_results, indent=2))