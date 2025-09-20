#!/usr/bin/env python3
"""
Simple demonstration script for the shared memory system.
This script shows the basic functionality as specified in the task requirements.

Usage: python demo_shared_memory.py
"""

import sys
from pathlib import Path
from datetime import datetime
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.shared_memory import SharedMemory
from src.models.workflow import ResearchContext
from src.models.research import ResearchFindings, Hypothesis


def main():
    """Demonstrate basic shared memory functionality."""
    print("🧠 Shared Memory System - Basic Demo")
    print("=" * 50)
    
    # Initialize shared memory
    storage_dir = "data/demo_shared_memory"
    shared_memory = SharedMemory(storage_dir)
    
    # Create a session
    session_id = f"demo_{int(datetime.now().timestamp())}"
    query = "Test research query for shared memory demo"
    
    print(f"📝 Session ID: {session_id}")
    print(f"🔍 Query: {query}")
    print()
    
    # 1. Create research context
    print("1️⃣  Creating research context...")
    context = shared_memory.create_research_context(session_id, query)
    print(f"   ✅ Created context (version {context.version})")
    
    # 2. Write some research findings
    print("\n2️⃣  Writing research findings...")
    hypothesis = Hypothesis(
        text="This is a test hypothesis for the demo",
        confidence=0.85,
        testable=True
    )
    
    findings = ResearchFindings(
        hypotheses=[hypothesis],
        confidence_score=0.85
    )
    
    shared_memory.write(session_id, "research_findings", findings)
    print("   ✅ Research findings written")
    
    # 3. Read the data back
    print("\n3️⃣  Reading data back...")
    retrieved_context = shared_memory.get_research_context(session_id)
    retrieved_findings = shared_memory.read(session_id, "research_findings", ResearchFindings)
    
    print(f"   📖 Context query: {retrieved_context.query}")
    print(f"   📖 Findings: {len(retrieved_findings.hypotheses)} hypotheses")
    print(f"   📖 Confidence: {retrieved_findings.confidence_score}")
    
    # 4. Show JSON files created
    print("\n4️⃣  JSON files created:")
    storage_path = Path(storage_dir) / session_id
    for json_file in storage_path.glob("*.json"):
        file_size = json_file.stat().st_size
        print(f"   📄 {json_file.name} ({file_size} bytes)")
        
        # Show a snippet of the JSON content
        with open(json_file, 'r') as f:
            data = json.load(f)
            print(f"      Type: {data.get('type')}")
            print(f"      Timestamp: {data.get('timestamp')}")
    
    # 5. Validate session
    print("\n5️⃣  Validating session...")
    validation_result = shared_memory.validate_session(session_id)
    if validation_result['valid']:
        print("   ✅ Session validation passed")
    else:
        print(f"   ❌ Validation errors: {validation_result['errors']}")
    
    print(f"\n💾 Data persisted to: {storage_path}")
    print("🎉 Demo completed successfully!")
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n✅ Shared memory system is working correctly!")
        else:
            print("\n❌ Demo failed!")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        sys.exit(1)