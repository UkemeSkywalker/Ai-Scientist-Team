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
            hypotheses: JSON string containing research hypotheses (optional - will read from shared memory)
            data_context: JSON string containing data metadata (optional - will read from shared memory)
            session_id: Session identifier for context management
            
        Returns:
            Dictionary containing experiment results
        """
        logger.info("Starting experiment execution", session_id=session_id)
        
        try:
            # Read Research Agent results from shared memory for hypotheses
            research_results = self.shared_memory.read(session_id, "research_result")
            if research_results:
                logger.info("Retrieved Research Agent results from shared memory", session_id=session_id)
                hypotheses_from_research = research_results.get("hypotheses", [])
                
                # CRITICAL FIX: Extract text from hypothesis objects
                processed_hypotheses = []
                for hyp in hypotheses_from_research:
                    if isinstance(hyp, dict) and 'text' in hyp:
                        processed_hypotheses.append(hyp['text'])
                    elif isinstance(hyp, str):
                        processed_hypotheses.append(hyp)
                    else:
                        processed_hypotheses.append(str(hyp))
                
                hypotheses_to_use = json.dumps({"hypotheses": processed_hypotheses})
                logger.info(f"Processed {len(processed_hypotheses)} hypotheses from Research Agent", session_id=session_id)
            else:
                logger.warning("No Research Agent results in shared memory, using provided hypotheses", session_id=session_id)
                hypotheses_to_use = hypotheses
            
            # Read Data Agent results from shared memory
            data_agent_results = self.shared_memory.read(session_id, "data_result")
            if data_agent_results:
                logger.info("Retrieved Data Agent results from shared memory", session_id=session_id)
                # Extract real data context with S3 paths from shared memory
                real_data_context = self._extract_data_context_from_shared_memory(data_agent_results)
                data_context_to_use = json.dumps(real_data_context)
            else:
                logger.warning("No Data Agent results in shared memory, using provided data_context", session_id=session_id)
                data_context_to_use = data_context
            
            if self.agent and STRANDS_AVAILABLE:
                try:
                    # Use Strands agent for experiments
                    result = await self._execute_strands_experiments(hypotheses_to_use, data_context_to_use, session_id)
                except Exception as strands_error:
                    logger.warning(f"Strands agent failed, falling back to direct tools: {str(strands_error)}")
                    # Fallback to direct tool execution if Strands fails
                    result = await self._execute_direct_experiments(hypotheses_to_use, data_context_to_use, session_id)
            else:
                # Fallback to direct tool execution
                result = await self._execute_direct_experiments(hypotheses_to_use, data_context_to_use, session_id)
            
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
                "hypotheses": hypotheses_to_use if 'hypotheses_to_use' in locals() else hypotheses,
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
        """Execute experiments using Strands agent with proper tool execution"""
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
            
            # Execute the conversation with the Strands agent using proper method
            logger.info("Starting Strands agent conversation", session_id=session_id)
            
            # Use the correct Strands agent execution method
            if hasattr(self.agent, '__call__'):
                response = self.agent(experiment_prompt)
            elif hasattr(self.agent, 'execute'):
                response = await self.agent.execute(experiment_prompt)
            else:
                raise AttributeError("Agent has no callable method")
            
            # Parse the structured results from Strands agent response
            parsed_results = self._parse_strands_response(response, session_id)
            
            if parsed_results:
                logger.info("Successfully parsed Strands agent response", session_id=session_id)
                return parsed_results
            else:
                logger.warning("Failed to parse Strands response, falling back to direct tools")
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
            
            # Step 2: Execute ML training with session_id for shared memory access
            logger.info("Executing ML training", session_id=session_id)
            training_results = sagemaker_training_tool(design_results, session_id)
            
            # Step 3: Perform statistical analysis
            logger.info("Performing statistical analysis", session_id=session_id)
            try:
                design_data = json.loads(design_results) if isinstance(design_results, str) else design_results
                training_data = json.loads(training_results) if isinstance(training_results, str) else training_results
                
                analysis_data = {
                    "config": design_data,
                    "training_results": training_data
                }
                analysis_results = statistical_analysis_tool(analysis_data)
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"JSON parsing error in statistical analysis: {str(e)}")
                analysis_results = statistical_analysis_tool({})
            
            # Step 4: Interpret results
            logger.info("Interpreting results", session_id=session_id)
            try:
                training_data = json.loads(training_results) if isinstance(training_results, str) else training_results
                analysis_data = json.loads(analysis_results) if isinstance(analysis_results, str) else analysis_results
                
                interpretation_data = {
                    "training_results": training_data,
                    "statistical_analysis": analysis_data
                }
                interpretation_results = results_interpretation_tool(interpretation_data)
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"JSON parsing error in results interpretation: {str(e)}")
                interpretation_results = results_interpretation_tool({})
            
            # Compile final results with safe JSON parsing
            try:
                final_results = {
                    "experiment_design": json.loads(design_results) if isinstance(design_results, str) else design_results,
                    "ml_training": json.loads(training_results) if isinstance(training_results, str) else training_results,
                    "statistical_analysis": json.loads(analysis_results) if isinstance(analysis_results, str) else analysis_results,
                    "interpretation": json.loads(interpretation_results) if isinstance(interpretation_results, str) else interpretation_results,
                    "execution_method": "direct_tools",
                    "session_id": session_id,
                    "status": "completed",
                    "timestamp": datetime.now().isoformat()
                }
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"JSON parsing error in final results: {str(e)}")
                final_results = {
                    "experiment_design": design_results,
                    "ml_training": training_results,
                    "statistical_analysis": analysis_results,
                    "interpretation": interpretation_results,
                    "execution_method": "direct_tools",
                    "session_id": session_id,
                    "status": "completed",
                    "timestamp": datetime.now().isoformat()
                }
            
            return final_results
            
        except Exception as e:
            logger.error(f"Direct experiment execution failed: {str(e)}")
            raise
    
    def _parse_strands_response(self, response, session_id: str) -> Optional[Dict[str, Any]]:
        """Parse Strands agent response to extract structured experiment results"""
        try:
            # Handle different response types from Strands agent
            if hasattr(response, 'content'):
                # Response with content attribute
                content = response.content
            elif hasattr(response, 'messages'):
                # Response with messages
                content = str(response.messages[-1]) if response.messages else str(response)
            else:
                # Direct string response
                content = str(response)
            
            logger.info(f"Parsing Strands response: {content[:200]}...", session_id=session_id)
            
            # Try to extract JSON from the response
            import re
            json_matches = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
            
            results = {
                "experiment_design": {},
                "ml_training": {},
                "statistical_analysis": {},
                "interpretation": {},
                "execution_method": "strands_agent",
                "session_id": session_id,
                "status": "completed",
                "timestamp": datetime.now().isoformat()
            }
            
            # Parse JSON results if found
            for json_str in json_matches:
                try:
                    parsed_json = json.loads(json_str)
                    if "experiments" in parsed_json:
                        results["experiment_design"] = parsed_json
                    elif "training_jobs" in parsed_json:
                        results["ml_training"] = parsed_json
                    elif "statistical_tests" in parsed_json:
                        results["statistical_analysis"] = parsed_json
                    elif "insights" in parsed_json:
                        results["interpretation"] = parsed_json
                except json.JSONDecodeError:
                    continue
            
            # If we have at least some structured results, return them
            if any(results[key] for key in ["experiment_design", "ml_training", "statistical_analysis", "interpretation"]):
                return results
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to parse Strands response: {str(e)}", session_id=session_id)
            return None
    
    def _extract_data_context_from_shared_memory(self, data_agent_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract data context with S3 paths from Data Agent results in shared memory"""
        processed_datasets = data_agent_results.get("processed_datasets", [])
        category = data_agent_results.get("category", "machine-learning")
        
        datasets = []
        for dataset in processed_datasets:
            original = dataset.get("original_dataset", {})
            storage = dataset.get("storage_results", {})
            cleaning = dataset.get("cleaning_results", {})
            
            # CRITICAL FIX: Extract S3 location and convert to proper string format
            s3_location = storage.get("s3_location", {})
            if isinstance(s3_location, dict) and s3_location.get('bucket') and s3_location.get('data_key'):
                bucket = s3_location.get('bucket')
                data_key = s3_location.get('data_key')
                s3_path = f"s3://{bucket}/{data_key}"  # Convert dict to string path
                logger.info(f"Converted S3 dict to path: {s3_path}")
            elif isinstance(s3_location, str) and s3_location.startswith('s3://'):
                s3_path = s3_location
                logger.info(f"Using existing S3 string path: {s3_path}")
            else:
                logger.warning(f"Invalid S3 location format for dataset {original.get('name')}: {s3_location}")
                continue  # Skip datasets without valid S3 paths
            
            # Determine task type and target
            dataset_name = original.get("name", "").lower()
            if "sentiment" in dataset_name or "review" in dataset_name:
                task_type = "classification"
                target_variable = "sentiment"
                columns = ["text_length", "sentiment_score", "rating", "word_count", "sentiment"]
            else:
                task_type = "classification"
                target_variable = "target"
                columns = ["feature_1", "feature_2", "feature_3", "feature_4", "target"]
            
            dataset_info = {
                "name": original.get("name", "unknown_dataset"),
                "type": "supervised",
                "task": task_type,
                "columns": columns,
                "target": target_variable,
                "s3_location": s3_path,  # Now guaranteed to be a string
                "quality_score": cleaning.get("quality_metrics", {}).get("overall_score", 0.8)
            }
            logger.info(f"Created dataset info with S3 path: {dataset_info['name']} -> {s3_path}")
            datasets.append(dataset_info)
        
        logger.info(f"Extracted {len(datasets)} datasets with valid S3 paths from shared memory")
        for i, dataset in enumerate(datasets):
            logger.info(f"Dataset {i+1}: {dataset['name']} -> {dataset['s3_location']}")
        
        return {
            "datasets": datasets,
            "data_summary": {
                "category": category,
                "total_samples": sum(1000 for _ in datasets),  # Estimated
                "source_diversity": len(set(d.get("source", "unknown") for d in processed_datasets)),
                "average_quality": sum(d.get("quality_score", 0) for d in datasets) / max(len(datasets), 1)
            }
        }
    
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