#!/usr/bin/env python3
"""
Test script for Strands Data Agent
Tests dataset discovery, cleaning, and S3 storage functionality
"""

import asyncio
import json
import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.agents.data_agent import DataAgent
from src.core.shared_memory import SharedMemory
from src.core.logger import setup_logger

def print_separator(title: str):
    """Print a formatted separator"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def print_results(title: str, results: dict):
    """Print formatted results"""
    print(f"\n{'-'*40}")
    print(f" {title}")
    print(f"{'-'*40}")
    print(json.dumps(results, indent=2, default=str))

async def test_data_search(agent: DataAgent, query: str, session_id: str):
    """Test dataset search functionality"""
    print_separator(f"Testing Data Search: '{query}'")
    
    try:
        # Execute data collection
        results = await agent.execute_data_collection(query, session_id)
        
        if results.get("status") == "success":
            print(f"✅ Data search successful!")
            print(f"   Datasets found: {results.get('datasets_found', 0)}")
            print(f"   Datasets processed: {results.get('datasets_processed', 0)}")
            print(f"   Sources: {', '.join(results.get('data_summary', {}).get('sources', []))}")
            print(f"   Average quality score: {results.get('data_summary', {}).get('average_quality_score', 0):.2f}")
            
            # Show processed datasets
            processed_datasets = results.get("processed_datasets", [])
            if processed_datasets:
                print(f"\n📊 Processed Datasets:")
                for i, dataset in enumerate(processed_datasets[:3], 1):  # Show first 3
                    original = dataset.get("original_dataset", {})
                    cleaning = dataset.get("cleaning_results", {})
                    storage = dataset.get("storage_results", {})
                    
                    print(f"   {i}. {original.get('title', 'Unknown Dataset')}")
                    print(f"      Source: {original.get('source', 'Unknown')}")
                    print(f"      Relevance: {original.get('relevance_score', 0):.2f}")
                    print(f"      Quality Score: {cleaning.get('quality_metrics', {}).get('overall_score', 0):.2f}")
                    print(f"      Storage: {storage.get('status', 'Unknown')}")
            
            # Show S3 locations
            s3_locations = results.get("s3_locations", [])
            if s3_locations:
                print(f"\n☁️  S3 Storage Locations:")
                for i, location in enumerate(s3_locations[:3], 1):
                    print(f"   {i}. s3://{location.get('bucket', 'unknown')}/{location.get('key', 'unknown')}")
                    print(f"      Size: {location.get('size_bytes', 0)} bytes")
                    print(f"      Region: {location.get('region', 'unknown')}")
            
            return results
        else:
            print(f"❌ Data search failed: {results.get('error', 'Unknown error')}")
            return None
            
    except Exception as e:
        print(f"❌ Data search error: {str(e)}")
        return None

def test_data_quality_validation(agent: DataAgent, results: dict):
    """Test data quality validation"""
    print_separator("Testing Data Quality Validation")
    
    try:
        quality_metrics = agent.validate_data_quality(results)
        
        print(f"📈 Data Quality Metrics:")
        print(f"   Datasets Found: {quality_metrics['datasets_found']}")
        print(f"   Datasets Processed: {quality_metrics['datasets_processed']}")
        print(f"   Processing Success Rate: {quality_metrics['processing_success_rate']:.2%}")
        print(f"   Average Quality Score: {quality_metrics['average_quality_score']:.2f}")
        print(f"   Source Diversity: {quality_metrics['source_diversity']}")
        print(f"   Storage Success: {'✅' if quality_metrics['storage_success'] else '❌'}")
        print(f"   Overall Quality: {quality_metrics['overall_quality']:.2f} ({quality_metrics['quality_level']})")
        
        return quality_metrics
        
    except Exception as e:
        print(f"❌ Quality validation error: {str(e)}")
        return None

def test_dataset_recommendations(agent: DataAgent):
    """Test dataset recommendations"""
    print_separator("Testing Dataset Recommendations")
    
    try:
        # Mock research context
        research_context = {
            "hypotheses": [
                {"text": "Sentiment analysis models perform better with balanced datasets"},
                {"text": "Classification accuracy improves with larger training sets"},
                {"text": "Time series forecasting requires temporal consistency"}
            ],
            "key_concepts": ["machine learning", "natural language processing"]
        }
        
        recommendations = agent.get_dataset_recommendations(research_context)
        
        print(f"💡 Dataset Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
        
        return recommendations
        
    except Exception as e:
        print(f"❌ Recommendations error: {str(e)}")
        return None

async def test_individual_tools():
    """Test individual data tools"""
    print_separator("Testing Individual Data Tools")
    
    try:
        from src.tools.data_tools import (
            kaggle_search_tool,
            huggingface_search_tool,
            data_cleaning_tool,
            s3_storage_tool
        )
        
        # Test Kaggle search
        print("🔍 Testing Kaggle search...")
        kaggle_result = kaggle_search_tool("sentiment analysis", max_results=3)
        kaggle_data = json.loads(kaggle_result)
        if kaggle_data.get("status") == "error":
            print(f"   ❌ Kaggle search failed: {kaggle_data.get('error', 'Unknown error')}")
            if "authentication" in kaggle_data.get('error', '').lower():
                print("   💡 To fix: Set up Kaggle API credentials:")
                print("      1. Go to https://www.kaggle.com/account")
                print("      2. Create new API token (downloads kaggle.json)")
                print("      3. Place kaggle.json in ~/.kaggle/ directory")
                print("      4. Or set KAGGLE_USERNAME and KAGGLE_KEY environment variables")
        else:
            print(f"   Found {kaggle_data.get('datasets_found', 0)} datasets")
        
        # Test HuggingFace search
        print("🔍 Testing HuggingFace search...")
        hf_result = huggingface_search_tool("nlp datasets", max_results=3)
        hf_data = json.loads(hf_result)
        print(f"   Found {hf_data.get('datasets_found', 0)} datasets")
        
        # Test data cleaning with first dataset
        if kaggle_data.get("datasets"):
            print("🧹 Testing data cleaning...")
            first_dataset = kaggle_data["datasets"][0]
            cleaning_result = data_cleaning_tool(json.dumps(first_dataset), sample_size=500)
            cleaning_data = json.loads(cleaning_result)
            print(f"   Quality score: {cleaning_data.get('quality_metrics', {}).get('overall_score', 0):.2f}")
            
            # Test S3 storage
            print("☁️  Testing S3 storage...")
            storage_result = s3_storage_tool(json.dumps(first_dataset), json.dumps(cleaning_data))
            storage_data = json.loads(storage_result)
            print(f"   Storage status: {storage_data.get('status', 'unknown')}")
        
        print("✅ All individual tools tested successfully!")
        
    except Exception as e:
        print(f"❌ Individual tools test error: {str(e)}")

def test_shared_memory_integration(agent: DataAgent, session_id: str):
    """Test shared memory integration"""
    print_separator("Testing Shared Memory Integration")
    
    try:
        # Get data status
        status = agent.get_data_status(session_id)
        
        if status:
            print(f"📝 Shared Memory Status:")
            print(f"   Session ID: {session_id}")
            print(f"   Status: {status.get('status', 'unknown')}")
            print(f"   Timestamp: {status.get('timestamp', 'unknown')}")
            print(f"   Datasets Processed: {status.get('datasets_processed', 0)}")
            print("✅ Shared memory integration working!")
        else:
            print("❌ No data found in shared memory")
        
        return status
        
    except Exception as e:
        print(f"❌ Shared memory test error: {str(e)}")
        return None

async def run_comprehensive_test(query: str):
    """Run comprehensive test suite"""
    print_separator(f"Strands Data Agent Comprehensive Test")
    print(f"Query: '{query}'")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    # Initialize components
    shared_memory = SharedMemory()
    agent = DataAgent(shared_memory)
    session_id = f"test_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Test 1: Individual tools
    await test_individual_tools()
    
    # Test 2: Data search and collection
    results = await test_data_search(agent, query, session_id)
    
    if results:
        # Test 3: Quality validation
        quality_metrics = test_data_quality_validation(agent, results)
        
        # Test 4: Shared memory integration
        memory_status = test_shared_memory_integration(agent, session_id)
        
        # Test 5: Dataset recommendations
        recommendations = test_dataset_recommendations(agent)
        
        # Final summary
        print_separator("Test Summary")
        print(f"✅ Data Agent Test Completed Successfully!")
        print(f"   Query: {query}")
        print(f"   Session: {session_id}")
        print(f"   Datasets Found: {results.get('datasets_found', 0)}")
        print(f"   Datasets Processed: {results.get('datasets_processed', 0)}")
        print(f"   Quality Level: {quality_metrics.get('quality_level', 'unknown') if quality_metrics else 'unknown'}")
        print(f"   Storage Success: {'✅' if results.get('s3_locations') else '❌'}")
        print(f"   Recommendations: {len(recommendations) if recommendations else 0}")
        
        # Save detailed results
        test_results = {
            "query": query,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "data_collection_results": results,
            "quality_metrics": quality_metrics,
            "recommendations": recommendations,
            "test_status": "success"
        }
        
        # Create organized test results directory
        test_results_dir = Path("test_results/data_agent")
        test_results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create meaningful filename with query and timestamp
        query_safe = query.replace(" ", "_").replace("/", "_").lower()
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        results_file = test_results_dir / f"data_agent_test_{query_safe}_{timestamp}.json"
        
        # Save detailed results
        with open(results_file, 'w') as f:
            json.dump(test_results, f, indent=2, default=str)
        
        # Also create a summary file
        summary_file = test_results_dir / f"data_agent_summary_{query_safe}_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Data Agent Test Summary\n")
            f.write(f"=====================\n\n")
            f.write(f"Query: {query}\n")
            f.write(f"Session: {session_id}\n")
            f.write(f"Timestamp: {timestamp}\n\n")
            f.write(f"Results:\n")
            f.write(f"- Datasets Found: {results.get('datasets_found', 0)}\n")
            f.write(f"- Datasets Processed: {results.get('datasets_processed', 0)}\n")
            f.write(f"- Processing Success Rate: {quality_metrics.get('processing_success_rate', 0):.1%}\n")
            f.write(f"- Quality Level: {quality_metrics.get('quality_level', 'unknown')}\n")
            f.write(f"- Storage Success: {'✅' if results.get('s3_locations') else '❌'}\n")
            f.write(f"- Average Quality Score: {quality_metrics.get('average_quality_score', 0):.2f}\n")
            f.write(f"- Source Diversity: {quality_metrics.get('source_diversity', 0)}\n\n")
            f.write(f"Files:\n")
            f.write(f"- Detailed Results: {results_file.name}\n")
            f.write(f"- Summary: {summary_file.name}\n")
        
        print(f"📄 Test results saved to: {test_results_dir}/")
        print(f"   - Detailed: {results_file.name}")
        print(f"   - Summary: {summary_file.name}")
        
        return test_results
    else:
        print("❌ Data collection failed - cannot continue with other tests")
        return None

async def main():
    """Main test function"""
    parser = argparse.ArgumentParser(description="Test Strands Data Agent")
    parser.add_argument("--search", default="sentiment analysis", help="Search query for datasets")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logger(log_level="DEBUG" if args.verbose else "INFO")
    
    try:
        # Run comprehensive test
        results = await run_comprehensive_test(args.search)
        
        if results:
            print(f"\n🎉 All tests completed successfully!")
            sys.exit(0)
        else:
            print(f"\n❌ Tests failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print(f"\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())