#!/usr/bin/env python3
"""
Test script for Strands SDK orchestrator and agent coordination
"""

import asyncio
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.orchestrator import StrandsOrchestrator
from src.core.shared_memory import SharedMemory
from src.core.logger import setup_logger, get_logger
from src.integrations.strands_config import strands_config

logger = get_logger(__name__)

async def test_strands_orchestrator(query: str, session_id: str = None) -> Dict[str, Any]:
    """
    Test the Strands orchestrator with a research query
    
    Args:
        query: Research query to process
        session_id: Optional session ID for testing
        
    Returns:
        Test results and workflow output
    """
    logger.info("Starting Strands orchestrator test", query=query)
    
    # Initialize components
    shared_memory = SharedMemory("data/test_shared_memory")
    orchestrator = StrandsOrchestrator(shared_memory)
    
    test_results = {
        "query": query,
        "session_id": session_id,
        "strands_available": False,
        "config_valid": False,
        "workflow_results": None,
        "errors": [],
        "success": False
    }
    
    try:
        # Test Strands SDK availability
        try:
            from strands import Agent
            test_results["strands_available"] = True
            logger.info("Strands SDK is available")
        except ImportError:
            logger.warning("Strands SDK not available, will use mock mode")
            test_results["strands_available"] = False
        
        # Test configuration
        test_results["config_valid"] = strands_config.validate_config()
        
        # Execute workflow
        logger.info("Executing workflow with Strands orchestrator")
        workflow_state = await orchestrator.execute_workflow(query, session_id)
        
        # Extract results
        test_results["workflow_results"] = {
            "session_id": workflow_state.session_id,
            "status": workflow_state.status if isinstance(workflow_state.status, str) else workflow_state.status.value,
            "start_time": workflow_state.start_time.isoformat() if workflow_state.start_time else None,
            "end_time": workflow_state.end_time.isoformat() if workflow_state.end_time else None,
            "current_agent": workflow_state.current_agent if isinstance(workflow_state.current_agent, str) else (workflow_state.current_agent.value if workflow_state.current_agent else None),
            "error": workflow_state.error
        }
        
        # Get agent results
        agent_results = {}
        for agent_type in workflow_state.agents:
            agent_status = workflow_state.agents[agent_type]
            agent_key = agent_type if isinstance(agent_type, str) else agent_type.value
            agent_results[agent_key] = {
                "status": agent_status.status if isinstance(agent_status.status, str) else agent_status.status.value,
                "progress": agent_status.progress,
                "results": agent_status.results,
                "error": agent_status.error
            }
        
        test_results["workflow_results"]["agents"] = agent_results
        
        # Get shared memory context
        context = shared_memory.get_strands_context(workflow_state.session_id)
        test_results["workflow_results"]["context_keys"] = list(context.keys())
        
        status_value = workflow_state.status if isinstance(workflow_state.status, str) else workflow_state.status.value
        test_results["success"] = status_value == "completed"
        
        logger.info("Strands orchestrator test completed", success=test_results["success"])
        
    except Exception as e:
        error_msg = f"Test failed: {str(e)}"
        logger.error("Strands orchestrator test failed", error=e)
        test_results["errors"].append(error_msg)
        test_results["success"] = False
    
    return test_results

async def test_agent_routing(orchestrator: StrandsOrchestrator) -> Dict[str, Any]:
    """Test agent routing and context sharing"""
    logger.info("Testing agent routing and context sharing")
    
    routing_tests = {
        "research_query": "machine learning bias detection",
        "data_query": "sentiment analysis datasets",
        "experiment_query": "classification model evaluation",
        "results": {},
        "success": True
    }
    
    try:
        for test_name, query in [
            ("research", routing_tests["research_query"]),
            ("data", routing_tests["data_query"]),
            ("experiment", routing_tests["experiment_query"])
        ]:
            logger.info(f"Testing {test_name} routing", query=query)
            
            workflow_state = await orchestrator.execute_workflow(
                query, 
                f"routing_test_{test_name}"
            )
            
            status_value = workflow_state.status if isinstance(workflow_state.status, str) else workflow_state.status.value
            routing_tests["results"][test_name] = {
                "status": status_value,
                "session_id": workflow_state.session_id,
                "success": status_value == "completed"
            }
            
            if status_value != "completed":
                routing_tests["success"] = False
    
    except Exception as e:
        logger.error("Agent routing test failed", error=e)
        routing_tests["success"] = False
        routing_tests["error"] = str(e)
    
    return routing_tests

def print_test_results(results: Dict[str, Any]):
    """Print formatted test results"""
    print("\n" + "="*60)
    print("STRANDS ORCHESTRATOR TEST RESULTS")
    print("="*60)
    
    print(f"Query: {results['query']}")
    print(f"Session ID: {results['session_id']}")
    print(f"Strands SDK Available: {'✅' if results['strands_available'] else '❌'}")
    print(f"Configuration Valid: {'✅' if results['config_valid'] else '❌'}")
    print(f"Overall Success: {'✅' if results['success'] else '❌'}")
    
    if results['errors']:
        print(f"\nErrors:")
        for error in results['errors']:
            print(f"  ❌ {error}")
    
    if results['workflow_results']:
        workflow = results['workflow_results']
        print(f"\nWorkflow Results:")
        print(f"  Status: {workflow['status']}")
        print(f"  Start Time: {workflow['start_time']}")
        print(f"  End Time: {workflow['end_time']}")
        print(f"  Current Agent: {workflow['current_agent']}")
        
        if workflow.get('agents'):
            print(f"\nAgent Results:")
            for agent_name, agent_data in workflow['agents'].items():
                status_icon = "✅" if agent_data['status'] == 'completed' else "❌"
                print(f"  {status_icon} {agent_name}: {agent_data['status']} ({agent_data['progress']}%)")
                if agent_data['error']:
                    print(f"    Error: {agent_data['error']}")
        
        if workflow.get('context_keys'):
            print(f"\nContext Keys: {', '.join(workflow['context_keys'])}")
    
    print("\n" + "="*60)

async def main():
    """Main test function"""
    parser = argparse.ArgumentParser(description="Test Strands orchestrator")
    parser.add_argument("--query", default="test research", help="Research query to test")
    parser.add_argument("--session-id", help="Session ID for testing")
    parser.add_argument("--test-routing", action="store_true", help="Test agent routing")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logger(log_level)
    
    print("🧪 Testing Strands SDK Orchestrator")
    print(f"Query: {args.query}")
    
    # Test basic orchestrator functionality
    results = await test_strands_orchestrator(args.query, args.session_id)
    print_test_results(results)
    
    # Test agent routing if requested
    if args.test_routing:
        print("\n🔄 Testing Agent Routing...")
        shared_memory = SharedMemory("data/test_shared_memory")
        orchestrator = StrandsOrchestrator(shared_memory)
        
        routing_results = await test_agent_routing(orchestrator)
        
        print(f"\nAgent Routing Test: {'✅' if routing_results['success'] else '❌'}")
        for test_name, test_result in routing_results['results'].items():
            status_icon = "✅" if test_result['success'] else "❌"
            print(f"  {status_icon} {test_name}: {test_result['status']}")
    
    # Summary
    if results['success']:
        print("\n🎉 All tests passed! Strands orchestrator is working correctly.")
        return 0
    else:
        print("\n❌ Some tests failed. Check the logs for details.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)