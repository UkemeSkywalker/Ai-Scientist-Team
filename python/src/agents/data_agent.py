"""
Strands Data Agent for the AI Scientist Team
Handles dataset discovery, collection, cleaning, and storage
"""

import json
import os
from typing import Dict, Any, Optional, List
from datetime import datetime
import structlog

try:
    from strands import Agent
    STRANDS_AVAILABLE = True
except ImportError:
    STRANDS_AVAILABLE = False
    Agent = None

from ..tools.data_tools import (
    kaggle_search_tool,
    huggingface_search_tool,
    data_cleaning_tool,
    s3_storage_tool,
    check_existing_datasets_tool,
    smart_dataset_discovery_tool,
    organize_dataset_categories_tool
)
from ..models.data import DataContext, DatasetMetadata, DataQualityMetrics, S3Location
from ..core.shared_memory import SharedMemory
from ..integrations.strands_config import strands_config

logger = structlog.get_logger(__name__)

class DataAgent:
    """
    Strands-powered Data Agent that handles dataset discovery, collection, and processing
    """
    
    def __init__(self, shared_memory: Optional[SharedMemory] = None):
        self.shared_memory = shared_memory or SharedMemory()
        self.agent = None
        self._initialize_agent()
    
    def _initialize_agent(self):
        """Initialize the Strands Data Agent with data collection tools"""
        if not STRANDS_AVAILABLE:
            logger.warning("Strands SDK not available, Data Agent will use direct tool calls")
            return
        
        try:
            # Validate Strands configuration
            if not strands_config.validate_config():
                logger.error("Strands configuration validation failed")
                return
            
            # Import conversation manager for better token management
            from strands.agent.conversation_manager import SlidingWindowConversationManager
            
            # Create Data Agent with specialized system prompt and enhanced tools
            self.agent = Agent(
                model=strands_config.model_id,
                system_prompt=self._get_system_prompt(),
                tools=[
                    check_existing_datasets_tool,
                    smart_dataset_discovery_tool,
                    kaggle_search_tool,
                    huggingface_search_tool,
                    data_cleaning_tool,
                    s3_storage_tool,
                    organize_dataset_categories_tool
                ],
                name="Data Agent",
                description="Specialized agent for intelligent dataset discovery, collection, cleaning, and category-based storage",
                agent_id="data_agent",
                conversation_manager=SlidingWindowConversationManager(window_size=15, should_truncate_results=True)
            )
            
            logger.info("Data Agent initialized successfully with Strands SDK")
            
        except Exception as e:
            logger.error(f"Failed to initialize Data Agent: {str(e)}")
            self.agent = None
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the Data Agent"""
        return """You are an intelligent Data Agent specialized in smart dataset discovery, collection, cleaning, and category-based storage.

Your enhanced workflow prioritizes reusability and organization:

1. SMART DISCOVERY FIRST:
   - Use smart_dataset_discovery_tool(query) to get comprehensive recommendations
   - This tool automatically checks existing datasets and searches for new ones if needed
   - Follow the recommendations to prioritize existing high-quality datasets

2. ALTERNATIVE INDIVIDUAL TOOLS (if needed):
   - check_existing_datasets_tool(query) - check what datasets we already have
   - kaggle_search_tool, huggingface_search_tool - search for new datasets
   - data_cleaning_tool - analyze and assess quality (requires dataset_info as JSON)
   - s3_storage_tool - store with category organization (requires dataset_info, cleaned_data_summary, and query)

3. ORGANIZATION TOOLS:
   - organize_dataset_categories_tool - reorganize existing datasets by research categories

KEY PRINCIPLES:
- Always check existing datasets before downloading new ones
- Store datasets in research categories (machine-learning, nlp, computer-vision, etc.)
- Pass complete JSON outputs between tools
- Prioritize reusability and avoid duplicate downloads
- Focus on building a well-organized, discoverable dataset library

The smart_dataset_discovery_tool is your primary tool - use it first for most queries."""

    async def execute_data_collection(self, query: str, session_id: str, research_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute comprehensive data collection workflow for the given query
        
        Args:
            query: Data search query (optional - will read from shared memory)
            session_id: Session identifier for context management
            research_context: Optional research context from previous agents
            
        Returns:
            Dictionary containing data collection results
        """
        logger.info("Starting data collection execution", query=query, session_id=session_id)
        
        try:
            # Read Research Agent results from shared memory
            research_results = self.shared_memory.read(session_id, "research_result")
            if research_results:
                logger.info("Retrieved Research Agent results from shared memory", session_id=session_id)
                query_from_research = research_results.get("query", query)
                research_context = research_results  # Use full research context
            else:
                logger.warning("No Research Agent results in shared memory, using provided query", session_id=session_id)
                query_from_research = query
            if self.agent and STRANDS_AVAILABLE:
                try:
                    # Use Strands agent for data collection
                    result = await self._execute_strands_data_collection(query_from_research, session_id, research_context)
                except Exception as strands_error:
                    logger.warning(f"Strands agent failed, falling back to direct tools: {str(strands_error)}")
                    # Fallback to direct tool execution if Strands fails
                    result = await self._execute_direct_data_collection(query_from_research, session_id, research_context)
            else:
                # Fallback to direct tool execution
                result = await self._execute_direct_data_collection(query_from_research, session_id, research_context)
            
            # Store results in shared memory
            self.shared_memory.update_context(session_id, {
                "data_result": result,
                "data_timestamp": datetime.now().isoformat()
            })
            
            logger.info("Data collection execution completed successfully", session_id=session_id)
            return result
            
        except Exception as e:
            error_msg = f"Data collection execution failed: {str(e)}"
            logger.error("Data collection execution failed", error=error_msg, session_id=session_id)
            
            error_result = {
                "query": query_from_research if 'query_from_research' in locals() else query,
                "error": error_msg,
                "status": "failed",
                "timestamp": datetime.now().isoformat()
            }
            
            # Store error in shared memory
            self.shared_memory.update_context(session_id, {
                "data_result": error_result,
                "data_error": error_msg
            })
            
            return error_result
    
    async def _execute_strands_data_collection(self, query: str, session_id: str, research_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute data collection using Strands agent with smart discovery workflow"""
        logger.info("Executing smart data collection with Strands agent", query=query)
        
        try:
            # Step 1: Use smart dataset discovery for comprehensive recommendations
            logger.info("Strands agent performing smart dataset discovery", query=query)
            smart_discovery_result = self.agent.tool.smart_dataset_discovery_tool(
                query=query, 
                max_new_datasets=5
            )
            
            # Parse smart discovery results
            if isinstance(smart_discovery_result, dict) and 'content' in smart_discovery_result:
                discovery_data = json.loads(smart_discovery_result['content'][0]['text'])
            else:
                discovery_data = json.loads(smart_discovery_result) if isinstance(smart_discovery_result, str) else smart_discovery_result
            
            # Extract recommendations
            recommendations = discovery_data.get("recommendations", [])
            category = discovery_data.get("category", "general")
            
            # Track all datasets found during the process
            all_datasets_found = []
            
            # Step 2: Process recommendations (prioritize existing datasets)
            processed_datasets = []
            s3_locations = []
            
            # Process existing datasets first (high priority)
            existing_datasets = [r for r in recommendations if r["type"] == "existing" and r["priority"] == "high"]
            for rec in existing_datasets[:3]:  # Top 3 existing
                dataset = rec["dataset"]
                all_datasets_found.append(dataset)  # Track found dataset
                try:
                    # For existing datasets, we already have them processed, just reference them
                    processed_dataset = {
                        "original_dataset": dataset,
                        "cleaning_results": {
                            "dataset_name": dataset.get("dataset_name", "unknown"),
                            "quality_metrics": {"overall_score": dataset.get("quality_score", 0.8)},
                            "status": "existing_dataset"
                        },
                        "storage_results": {
                            "dataset_name": dataset.get("dataset_name", "unknown"),
                            "s3_location": {
                                "bucket": os.getenv("S3_BUCKET_NAME", "ai-scientist-team-data"),
                                "key": dataset.get("s3_key", ""),
                                "category_path": f"datasets/{category}/",
                                "region": "us-east-1",
                                "size_bytes": dataset.get("size_bytes", 0)
                            },
                            "status": "existing_dataset"
                        }
                    }
                    processed_datasets.append(processed_dataset)
                    s3_locations.append(processed_dataset["storage_results"]["s3_location"])
                    
                except Exception as e:
                    logger.warning(f"Failed to process existing dataset {dataset.get('dataset_name')}: {str(e)}")
                    continue
            
            # Process new datasets if needed
            new_datasets = [r for r in recommendations if r["type"] == "new"]
            for rec in new_datasets[:4]:  # Top 4 new datasets
                dataset = rec["dataset"]
                all_datasets_found.append(dataset)  # Track found dataset
                try:
                    logger.info("Processing new dataset", dataset_name=dataset.get("name"))
                    
                    # Clean the dataset
                    from ..tools.data_tools import data_cleaning_tool
                    cleaning_result_str = data_cleaning_tool(
                        json.dumps(dataset),
                        sample_size=1000
                    )
                    cleaning_result = json.loads(cleaning_result_str)
                    
                    # Store in S3 with category information
                    from ..tools.data_tools import s3_storage_tool
                    storage_result_str = s3_storage_tool(
                        json.dumps(dataset),
                        json.dumps(cleaning_result),  # Pass full cleaning result with DataFrame
                        query=query  # Pass query for categorization
                    )
                    storage_data = json.loads(storage_result_str)
                    
                    # Extract metadata for compatibility
                    cleaning_data = cleaning_result.get("metadata", cleaning_result)
                    
                    # Combine results
                    processed_dataset = {
                        "original_dataset": dataset,
                        "cleaning_results": cleaning_data,
                        "storage_results": storage_data
                    }
                    processed_datasets.append(processed_dataset)
                    
                    if storage_data.get("s3_location"):
                        s3_locations.append(storage_data["s3_location"])
                    
                except Exception as dataset_error:
                    logger.warning(f"Failed to process new dataset {dataset.get('name')}: {str(dataset_error)}")
                    continue
            
            # Fallback: if we don't have enough datasets, search manually
            if len(processed_datasets) < 7:  # Increase threshold to get more datasets
                logger.info("Insufficient datasets from smart discovery, performing additional searches")
                
                # Search Kaggle and HuggingFace as fallback with higher limits
                kaggle_result = self.agent.tool.kaggle_search_tool(query=query, max_results=10)
                huggingface_result = self.agent.tool.huggingface_search_tool(query=query, max_results=8)
            
                # Parse fallback search results
                if isinstance(kaggle_result, dict) and 'content' in kaggle_result:
                    kaggle_data = json.loads(kaggle_result['content'][0]['text'])
                else:
                    kaggle_data = json.loads(kaggle_result) if isinstance(kaggle_result, str) else kaggle_result
                    
                if isinstance(huggingface_result, dict) and 'content' in huggingface_result:
                    huggingface_data = json.loads(huggingface_result['content'][0]['text'])
                else:
                    huggingface_data = json.loads(huggingface_result) if isinstance(huggingface_result, str) else huggingface_result
                
                # Process additional datasets
                additional_datasets = []
                if kaggle_data.get("datasets"):
                    additional_datasets.extend(kaggle_data["datasets"][:8])  # Take more from Kaggle
                if huggingface_data.get("datasets"):
                    additional_datasets.extend(huggingface_data["datasets"][:6])  # Take more from HuggingFace
                
                for dataset in additional_datasets:
                    if len(processed_datasets) >= 15:  # Increase total limit to 15
                        break
                    
                    all_datasets_found.append(dataset)  # Track found dataset
                    try:
                        # Clean and store additional datasets
                        from ..tools.data_tools import data_cleaning_tool, s3_storage_tool
                        
                        cleaning_result_str = data_cleaning_tool(json.dumps(dataset), sample_size=1000)
                        cleaning_result = json.loads(cleaning_result_str)
                        
                        storage_result_str = s3_storage_tool(
                            json.dumps(dataset),
                            json.dumps(cleaning_result),  # Pass full cleaning result
                            query=query
                        )
                        storage_data = json.loads(storage_result_str)
                        
                        # Extract metadata for compatibility
                        cleaning_data = cleaning_result.get("metadata", cleaning_result)
                        
                        processed_dataset = {
                            "original_dataset": dataset,
                            "cleaning_results": cleaning_data,
                            "storage_results": storage_data
                        }
                        processed_datasets.append(processed_dataset)
                        
                        if storage_data.get("s3_location"):
                            s3_locations.append(storage_data["s3_location"])
                            
                    except Exception as e:
                        logger.warning(f"Failed to process additional dataset {dataset.get('name')}: {str(e)}")
                        continue
            
            # Create enhanced final results with smart discovery information
            final_results = {
                "query": query,
                "session_id": session_id,
                "execution_method": "strands_agent_smart_discovery",
                "smart_discovery": discovery_data,
                "category": category,
                "datasets_found": len(all_datasets_found),  # Total unique datasets found
                "datasets_processed": len(processed_datasets),
                "processed_datasets": processed_datasets,
                "s3_locations": s3_locations,
                "data_summary": {
                    "total_datasets": len(processed_datasets),
                    "existing_datasets_reused": len([d for d in processed_datasets if d["storage_results"].get("status") == "existing_dataset"]),
                    "new_datasets_added": len([d for d in processed_datasets if d["storage_results"].get("status") not in ["existing_dataset"]]),
                    "category": category,
                    "sources": list(set(d["original_dataset"].get("source", "unknown") for d in processed_datasets)),
                    "average_quality_score": sum(d["cleaning_results"].get("quality_metrics", {}).get("overall_score", 0) 
                                                for d in processed_datasets) / max(len(processed_datasets), 1),
                    "total_storage_size": sum(loc.get("size_bytes", 0) for loc in s3_locations),
                    "reusability_achieved": len([d for d in processed_datasets if d["storage_results"].get("status") == "existing_dataset"]) > 0
                },
                "organization_benefits": {
                    "category_based_storage": True,
                    "future_reusability": True,
                    "discoverable_structure": True,
                    "category_keywords": discovery_data.get("category", "general")
                },
                "status": "success",
                "timestamp": datetime.now().isoformat()
            }
            
            return final_results
            
        except Exception as e:
            logger.error(f"Strands data collection execution failed: {str(e)}")
            # Fallback to direct execution
            return await self._execute_direct_data_collection(query, session_id, research_context)
    
    async def _execute_direct_data_collection(self, query: str, session_id: str, research_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute data collection using direct tool calls with smart discovery (fallback)"""
        logger.info("Executing smart data collection with direct tool calls", query=query)
        
        try:
            # Step 1: Use smart dataset discovery
            logger.info("Performing smart dataset discovery", query=query)
            smart_discovery_results = smart_dataset_discovery_tool(query, max_new_datasets=5)
            discovery_data = json.loads(smart_discovery_results)
            
            # Extract recommendations and category
            recommendations = discovery_data.get("recommendations", [])
            category = discovery_data.get("category", "general")
            
            # Track all datasets found during the process
            all_datasets_found = []
            
            # Step 2: Process recommendations
            processed_datasets = []
            s3_locations = []
            
            # Process existing datasets first
            existing_datasets = [r for r in recommendations if r["type"] == "existing"]
            for rec in existing_datasets[:3]:  # Top 3 existing
                dataset = rec["dataset"]
                all_datasets_found.append(dataset)  # Track found dataset
                try:
                    processed_dataset = {
                        "original_dataset": dataset,
                        "cleaning_results": {
                            "dataset_name": dataset.get("dataset_name", "unknown"),
                            "quality_metrics": {"overall_score": dataset.get("quality_score", 0.8)},
                            "status": "existing_dataset"
                        },
                        "storage_results": {
                            "dataset_name": dataset.get("dataset_name", "unknown"),
                            "s3_location": {
                                "bucket": os.getenv("S3_BUCKET_NAME", "ai-scientist-team-data"),
                                "key": dataset.get("s3_key", ""),
                                "category_path": f"datasets/{category}/",
                                "region": "us-east-1",
                                "size_bytes": dataset.get("size_bytes", 0)
                            },
                            "status": "existing_dataset"
                        }
                    }
                    processed_datasets.append(processed_dataset)
                    s3_locations.append(processed_dataset["storage_results"]["s3_location"])
                    
                except Exception as e:
                    logger.warning(f"Failed to process existing dataset {dataset.get('dataset_name')}: {str(e)}")
                    continue
            
            # Process new datasets
            new_datasets = [r for r in recommendations if r["type"] == "new"]
            for rec in new_datasets[:4]:  # Top 4 new datasets
                dataset = rec["dataset"]
                all_datasets_found.append(dataset)  # Track found dataset
                try:
                    logger.info("Processing new dataset", dataset_name=dataset.get("name"))
                    
                    # Clean the dataset
                    cleaning_results = data_cleaning_tool(json.dumps(dataset), sample_size=1000)
                    cleaning_result = json.loads(cleaning_results)
                    
                    # Store in S3 with category information
                    storage_results = s3_storage_tool(
                        json.dumps(dataset), 
                        json.dumps(cleaning_result),  # Pass full cleaning result
                        query=query  # Pass query for categorization
                    )
                    storage_data = json.loads(storage_results)
                    
                    # Extract metadata for compatibility
                    cleaning_data = cleaning_result.get("metadata", cleaning_result)
                    
                    # Combine results
                    processed_dataset = {
                        "original_dataset": dataset,
                        "cleaning_results": cleaning_data,
                        "storage_results": storage_data
                    }
                    processed_datasets.append(processed_dataset)
                    
                    if storage_data.get("s3_location"):
                        s3_locations.append(storage_data["s3_location"])
                    
                except Exception as dataset_error:
                    logger.warning(f"Failed to process new dataset {dataset.get('name')}: {str(dataset_error)}")
                    continue
            
            # Fallback: if we don't have enough datasets, search manually
            if len(processed_datasets) < 7:  # Increase threshold
                logger.info("Insufficient datasets from smart discovery, performing additional searches")
                
                kaggle_results = kaggle_search_tool(query, max_results=10)  # Increase search limits
                huggingface_results = huggingface_search_tool(query, max_results=8)
                
                kaggle_data = json.loads(kaggle_results)
                huggingface_data = json.loads(huggingface_results)
                
                additional_datasets = []
                if kaggle_data.get("datasets"):
                    additional_datasets.extend(kaggle_data["datasets"][:8])  # Take more datasets
                if huggingface_data.get("datasets"):
                    additional_datasets.extend(huggingface_data["datasets"][:6])
                
                for dataset in additional_datasets:
                    if len(processed_datasets) >= 15:  # Increase total limit
                        break
                    
                    all_datasets_found.append(dataset)  # Track found dataset
                    try:
                        cleaning_results = data_cleaning_tool(json.dumps(dataset), sample_size=1000)
                        cleaning_result = json.loads(cleaning_results)
                        
                        storage_results = s3_storage_tool(
                            json.dumps(dataset), 
                            json.dumps(cleaning_result),  # Pass full cleaning result
                            query=query
                        )
                        storage_data = json.loads(storage_results)
                        
                        # Extract metadata for compatibility
                        cleaning_data = cleaning_result.get("metadata", cleaning_result)
                        
                        processed_dataset = {
                            "original_dataset": dataset,
                            "cleaning_results": cleaning_data,
                            "storage_results": storage_data
                        }
                        processed_datasets.append(processed_dataset)
                        
                        if storage_data.get("s3_location"):
                            s3_locations.append(storage_data["s3_location"])
                            
                    except Exception as e:
                        logger.warning(f"Failed to process additional dataset {dataset.get('name')}: {str(e)}")
                        continue
            
            # Create enhanced final results
            final_results = {
                "query": query,
                "session_id": session_id,
                "execution_method": "direct_tools_smart_discovery",
                "smart_discovery": discovery_data,
                "category": category,
                "datasets_found": len(all_datasets_found),  # Total unique datasets found
                "datasets_processed": len(processed_datasets),
                "processed_datasets": processed_datasets,
                "s3_locations": s3_locations,
                "data_summary": {
                    "total_datasets": len(processed_datasets),
                    "existing_datasets_reused": len([d for d in processed_datasets if d["storage_results"].get("status") == "existing_dataset"]),
                    "new_datasets_added": len([d for d in processed_datasets if d["storage_results"].get("status") not in ["existing_dataset"]]),
                    "category": category,
                    "sources": list(set(d["original_dataset"].get("source", "unknown") for d in processed_datasets)),
                    "average_quality_score": sum(d["cleaning_results"].get("quality_metrics", {}).get("overall_score", 0) 
                                                for d in processed_datasets) / max(len(processed_datasets), 1),
                    "total_storage_size": sum(loc.get("size_bytes", 0) for loc in s3_locations),
                    "reusability_achieved": len([d for d in processed_datasets if d["storage_results"].get("status") == "existing_dataset"]) > 0
                },
                "organization_benefits": {
                    "category_based_storage": True,
                    "future_reusability": True,
                    "discoverable_structure": True,
                    "category_keywords": category
                },
                "status": "success",
                "timestamp": datetime.now().isoformat()
            }
            
            return final_results
            
        except Exception as e:
            logger.error(f"Direct data collection execution failed: {str(e)}")
            raise
    
    def get_data_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get the current data collection status for a session"""
        context = self.shared_memory.get_context(session_id)
        if context and "data_result" in context:
            return context["data_result"]
        return None
    
    def validate_data_quality(self, data_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the quality of data collection results"""
        quality_metrics = {
            "datasets_found": data_results.get("datasets_found", 0),
            "datasets_processed": data_results.get("datasets_processed", 0),
            "processing_success_rate": 0,
            "average_quality_score": 0,
            "source_diversity": 0,
            "storage_success": False
        }
        
        # Calculate processing success rate
        if quality_metrics["datasets_found"] > 0:
            quality_metrics["processing_success_rate"] = quality_metrics["datasets_processed"] / quality_metrics["datasets_found"]
        
        # Get average quality score
        data_summary = data_results.get("data_summary", {})
        quality_metrics["average_quality_score"] = data_summary.get("average_quality_score", 0)
        
        # Count source diversity
        sources = data_summary.get("sources", [])
        quality_metrics["source_diversity"] = len(sources)
        
        # Check storage success
        s3_locations = data_results.get("s3_locations", [])
        quality_metrics["storage_success"] = len(s3_locations) > 0
        
        # Calculate overall quality score with more realistic thresholds
        quality_score = 0
        
        # Processing success rate scoring (more lenient)
        if quality_metrics["processing_success_rate"] >= 0.8:
            quality_score += 0.3
        elif quality_metrics["processing_success_rate"] >= 0.5:
            quality_score += 0.25
        elif quality_metrics["processing_success_rate"] >= 0.2:
            quality_score += 0.2
        elif quality_metrics["processing_success_rate"] > 0:
            quality_score += 0.1
        
        # Data quality scoring
        if quality_metrics["average_quality_score"] >= 0.9:
            quality_score += 0.3
        elif quality_metrics["average_quality_score"] >= 0.7:
            quality_score += 0.25
        elif quality_metrics["average_quality_score"] >= 0.5:
            quality_score += 0.2
        
        # Source diversity scoring
        if quality_metrics["source_diversity"] >= 2:
            quality_score += 0.2
        elif quality_metrics["source_diversity"] >= 1:
            quality_score += 0.15
        
        # Storage success scoring
        if quality_metrics["storage_success"]:
            quality_score += 0.2
        
        quality_metrics["overall_quality"] = round(quality_score, 2)
        quality_metrics["quality_level"] = (
            "high" if quality_score >= 0.8 else
            "medium" if quality_score >= 0.5 else
            "low"
        )
        
        return quality_metrics
    
    def get_dataset_recommendations(self, research_context: Dict[str, Any]) -> List[str]:
        """Get dataset recommendations based on research context"""
        recommendations = []
        
        # Extract key concepts from research context
        hypotheses = research_context.get("hypotheses", [])
        key_concepts = research_context.get("key_concepts", [])
        
        # Generate recommendations based on hypotheses
        for hypothesis in hypotheses[:3]:  # Top 3 hypotheses
            hypothesis_text = hypothesis.get("text", "")
            if "sentiment" in hypothesis_text.lower():
                recommendations.append("Search for sentiment analysis datasets with labeled emotions")
            elif "classification" in hypothesis_text.lower():
                recommendations.append("Look for multi-class classification datasets")
            elif "time series" in hypothesis_text.lower():
                recommendations.append("Find temporal datasets with time-based patterns")
            else:
                recommendations.append(f"Search for datasets related to: {hypothesis_text[:50]}...")
        
        # Add recommendations based on key concepts
        for concept in key_concepts[:2]:  # Top 2 concepts
            recommendations.append(f"Find datasets containing '{concept}' for validation")
        
        return recommendations[:5]  # Limit to 5 recommendations

# Create a convenience function for easy agent creation
def create_data_agent(shared_memory: Optional[SharedMemory] = None) -> DataAgent:
    """Create and return a Data Agent instance"""
    return DataAgent(shared_memory)