#!/usr/bin/env python3
"""
Test script for smart dataset discovery and category-based organization
Demonstrates the enhanced data tools with intelligent dataset management
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add the parent directory to Python path to enable imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(parent_dir / "src"))

# Import the modules
import importlib.util

# Load data_tools module
spec = importlib.util.spec_from_file_location("data_tools", parent_dir / "src" / "tools" / "data_tools.py")
data_tools = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_tools)

# Load data_agent module  
spec = importlib.util.spec_from_file_location("data_agent", parent_dir / "src" / "agents" / "data_agent.py")
data_agent_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_agent_module)

# Load shared_memory module
spec = importlib.util.spec_from_file_location("shared_memory", parent_dir / "src" / "core" / "shared_memory.py")
shared_memory_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared_memory_module)

# Extract the functions and classes we need
categorize_query = data_tools.categorize_query
check_existing_datasets_tool = data_tools.check_existing_datasets_tool
smart_dataset_discovery_tool = data_tools.smart_dataset_discovery_tool
organize_dataset_categories_tool = data_tools.organize_dataset_categories_tool
DataAgent = data_agent_module.DataAgent
SharedMemory = shared_memory_module.SharedMemory

def test_query_categorization():
    """Test the query categorization functionality"""
    print("=" * 60)
    print("TESTING QUERY CATEGORIZATION")
    print("=" * 60)
    
    test_queries = [
        "sentiment analysis of movie reviews",
        "computer vision object detection datasets",
        "natural language processing for chatbots",
        "machine learning classification algorithms",
        "healthcare patient diagnosis data",
        "financial stock market prediction",
        "climate change temperature data",
        "social media user engagement analysis"
    ]
    
    for query in test_queries:
        category, confidence = categorize_query(query)
        print(f"Query: '{query}'")
        print(f"  → Category: {category} (confidence: {confidence:.2f})")
        print()

def test_existing_datasets_check():
    """Test checking existing datasets in S3"""
    print("=" * 60)
    print("TESTING EXISTING DATASETS CHECK")
    print("=" * 60)
    
    test_query = "machine learning sentiment analysis"
    print(f"Checking existing datasets for: '{test_query}'")
    
    result = check_existing_datasets_tool(test_query)
    data = json.loads(result)
    
    print(f"Primary Category: {data.get('primary_category')}")
    print(f"Category Confidence: {data.get('category_confidence')}")
    print(f"Recommendation: {data.get('summary', {}).get('recommendation')}")
    print(f"Status: {data.get('status')}")
    print()

def test_smart_dataset_discovery():
    """Test the smart dataset discovery tool"""
    print("=" * 60)
    print("TESTING SMART DATASET DISCOVERY")
    print("=" * 60)
    
    test_queries = [
        "sentiment analysis movie reviews",
        "computer vision image classification",
        "natural language processing text analysis"
    ]
    
    for query in test_queries:
        print(f"Smart discovery for: '{query}'")
        
        result = smart_dataset_discovery_tool(query, max_new_datasets=3)
        data = json.loads(result)
        
        print(f"  Category: {data.get('category')}")
        print(f"  Strategy: {data.get('strategy', {}).get('primary_approach')}")
        print(f"  Recommendations: {len(data.get('recommendations', []))}")
        
        # Show top recommendations
        for i, rec in enumerate(data.get('recommendations', [])[:3]):
            print(f"    {i+1}. {rec.get('type')} - {rec.get('priority')} priority")
            print(f"       Dataset: {rec.get('dataset', {}).get('name', 'Unknown')}")
            print(f"       Action: {rec.get('action')}")
        
        print(f"  Status: {data.get('status')}")
        print()

def test_dataset_organization():
    """Test dataset organization tool"""
    print("=" * 60)
    print("TESTING DATASET ORGANIZATION")
    print("=" * 60)
    
    print("Testing dataset reorganization (dry run)...")
    
    result = organize_dataset_categories_tool(dry_run=True)
    data = json.loads(result)
    
    print(f"Datasets analyzed: {data.get('summary', {}).get('total_datasets_analyzed', 0)}")
    print(f"Datasets needing reorganization: {data.get('summary', {}).get('datasets_needing_reorganization', 0)}")
    print(f"Category distribution: {data.get('summary', {}).get('category_distribution', {})}")
    print(f"Status: {data.get('status')}")
    print()

async def test_enhanced_data_agent():
    """Test the enhanced data agent with smart discovery"""
    print("=" * 60)
    print("TESTING ENHANCED DATA AGENT")
    print("=" * 60)
    
    # Initialize shared memory and data agent
    shared_memory = SharedMemory()
    data_agent = DataAgent(shared_memory)
    
    test_query = "sentiment analysis of social media posts"
    session_id = "test_smart_discovery_001"
    
    print(f"Testing data agent with query: '{test_query}'")
    print("This will demonstrate:")
    print("- Smart dataset discovery")
    print("- Category-based organization")
    print("- Reusability prioritization")
    print()
    
    try:
        # Execute data collection with smart discovery
        result = await data_agent.execute_data_collection(
            query=test_query,
            session_id=session_id
        )
        
        print("Data Collection Results:")
        print(f"  Execution Method: {result.get('execution_method')}")
        print(f"  Category: {result.get('category')}")
        print(f"  Datasets Found: {result.get('datasets_found')}")
        print(f"  Datasets Processed: {result.get('datasets_processed')}")
        
        data_summary = result.get('data_summary', {})
        print(f"  Existing Datasets Reused: {data_summary.get('existing_datasets_reused', 0)}")
        print(f"  New Datasets Added: {data_summary.get('new_datasets_added', 0)}")
        print(f"  Average Quality Score: {data_summary.get('average_quality_score', 0):.2f}")
        print(f"  Reusability Achieved: {data_summary.get('reusability_achieved', False)}")
        
        organization_benefits = result.get('organization_benefits', {})
        print(f"  Category-Based Storage: {organization_benefits.get('category_based_storage', False)}")
        print(f"  Future Reusability: {organization_benefits.get('future_reusability', False)}")
        
        print(f"  Status: {result.get('status')}")
        
        # Validate data quality
        quality_metrics = data_agent.validate_data_quality(result)
        print(f"\nQuality Assessment:")
        print(f"  Overall Quality: {quality_metrics.get('overall_quality')}")
        print(f"  Quality Level: {quality_metrics.get('quality_level')}")
        print(f"  Processing Success Rate: {quality_metrics.get('processing_success_rate'):.2f}")
        print(f"  Source Diversity: {quality_metrics.get('source_diversity')}")
        
    except Exception as e:
        print(f"Error during data agent test: {str(e)}")
    
    print()

def demonstrate_benefits():
    """Demonstrate the benefits of the new system"""
    print("=" * 60)
    print("SMART DATASET DISCOVERY BENEFITS")
    print("=" * 60)
    
    benefits = [
        "🎯 INTELLIGENT CATEGORIZATION",
        "   • Automatically categorizes queries into research domains",
        "   • Organizes datasets by topic (ML, NLP, Computer Vision, etc.)",
        "   • Enables easy discovery and reuse",
        "",
        "♻️  REUSABILITY FIRST",
        "   • Checks existing datasets before downloading new ones",
        "   • Prioritizes high-quality existing data",
        "   • Reduces redundant downloads and storage costs",
        "",
        "📁 ORGANIZED STORAGE",
        "   • Category-based S3 folder structure",
        "   • Consistent naming and metadata",
        "   • Easy browsing and management",
        "",
        "🔍 SMART RECOMMENDATIONS",
        "   • Combines existing and new dataset suggestions",
        "   • Relevance-based prioritization",
        "   • Quality-aware selection",
        "",
        "🚀 EFFICIENCY GAINS",
        "   • Faster research iterations",
        "   • Better dataset coverage",
        "   • Reduced manual dataset hunting"
    ]
    
    for benefit in benefits:
        print(benefit)
    
    print()

async def main():
    """Run all tests and demonstrations"""
    print("🤖 AI SCIENTIST TEAM - SMART DATASET DISCOVERY")
    print("Enhanced with Category-Based Organization & Intelligent Reuse")
    print()
    
    # Run individual tool tests
    test_query_categorization()
    test_existing_datasets_check()
    test_smart_dataset_discovery()
    test_dataset_organization()
    
    # Run integrated data agent test
    await test_enhanced_data_agent()
    
    # Show benefits
    demonstrate_benefits()
    
    print("✅ All tests completed!")
    print("\nThe smart dataset discovery system is ready to:")
    print("1. Automatically categorize research queries")
    print("2. Check existing datasets before downloading new ones")
    print("3. Organize datasets in category-based S3 structure")
    print("4. Provide intelligent recommendations for dataset reuse")
    print("5. Build a reusable, well-organized dataset library")

if __name__ == "__main__":
    # Set up environment
    os.environ.setdefault("S3_BUCKET_NAME", "ai-scientist-team-data")
    os.environ.setdefault("AWS_REGION", "us-east-1")
    
    # Run the tests
    asyncio.run(main())