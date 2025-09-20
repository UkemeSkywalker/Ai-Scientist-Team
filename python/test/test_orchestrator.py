#!/usr/bin/env python3
"""
Test script for the workflow orchestrator.
Usage: python test_orchestrator.py --query "Your research question"
"""

import asyncio
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.orchestrator import WorkflowOrchestrator
from src.core.shared_memory import SharedMemory


async def main():
    parser = argparse.ArgumentParser(description="Test the AI Scientist Team orchestrator")
    parser.add_argument(
        "--query", 
        default="How can machine learning bias be reduced in healthcare applications?",
        help="Research query to test"
    )
    parser.add_argument(
        "--mock-agents", 
        action="store_true",
        help="Use mock agents for testing"
    )
    
    args = parser.parse_args()
    
    print(f"🚀 Starting AI Scientist Team test with query: {args.query}")
    print("=" * 80)
    
    # Initialize orchestrator
    shared_memory = SharedMemory("data/test_shared_memory")
    orchestrator = WorkflowOrchestrator(shared_memory)
    
    try:
        # Execute workflow
        workflow_state = await orchestrator.execute_workflow(args.query)
        
        print(f"\n✅ Workflow completed successfully!")
        print(f"Session ID: {workflow_state.session_id}")
        print(f"Status: {workflow_state.status}")
        print(f"Duration: {(workflow_state.end_time - workflow_state.start_time).total_seconds():.1f}s")
        
        # Print agent results
        print("\n📊 Agent Results:")
        print("-" * 40)
        
        for agent_type, agent_status in workflow_state.agents.items():
            print(f"\n{str(agent_type).upper()} AGENT:")
            print(f"  Status: {agent_status.status}")
            print(f"  Progress: {agent_status.progress}%")
            if agent_status.results:
                print(f"  Results: {agent_status.results}")
        
    except Exception as e:
        print(f"❌ Workflow failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)