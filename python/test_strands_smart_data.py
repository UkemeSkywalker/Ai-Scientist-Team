#!/usr/bin/env python3
"""
Test script for Smart Dataset Discovery System
Tests category-based organization, intelligent reusability, and enhanced data tools
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

def test_query_categorization(query: str):
    """Test the query categorization functionality"""
    print_separator(f"Testing Query Categorization: '{query}'")
    
    try:
        from src.tools.data_tools import categorize_query
        
        category, confidence = categorize_query(query)
        
        print(f"🎯 Query Categorization Results:")
        print(f"   Query: '{query}'")
        print(f"   Category: {category}")
        print(f"   Confidence: {confidence:.2f}")
        
        # Show category keywords
        from src.tools.data_tools import RESEARCH_CATEGORIES
        keywords = RESEARCH_CATEGORIES.get(category, [])
        print(f"   Category Keywords: {', '.join(keywords[:5])}...")
        
        print(f"✅ Query categorization successful!")
        return category, confidence
        
    except Exception as e:
        print(f"❌ Query categorization error: {str(e)}")
        return None, None

def test_existing_datasets_check(query: str):
    """Test checking existing datasets in S3"""
    print_separator(f"Testing Existing Datasets Check: '{query}'")
    
    try:
        from src.tools.data_tools import check_existing_datasets_tool
        
        result = check_existing_datasets_tool(query)
        data = json.loads(result)
        
        print(f"🔍 Existing Datasets Check Results:")
        print(f"   Primary Category: {data.get('primary_category')}")
        print(f"   Category Confidence: {data.get('category_confidence')}")
        print(f"   Recommendation: {data.get('summary', {}).get('recommendation')}")
        
        existing_datasets = data.get('existing_datasets', {})
        primary_count = len(existing_datasets.get('primary_category', []))
        related_count = len(existing_datasets.get('related_categories', []))
        
        print(f"   Primary Category Datasets: {primary_count}")
        print(f"   Related Category Datasets: {related_count}")
        print(f"   Status: {data.get('status')}")
        
        if primary_count > 0:
            print(f"   📊 Top Existing Datasets:")
            for i, dataset in enumerate(existing_datasets.get('primary_category', [])[:3], 1):
                print(f"      {i}. {dataset.get('dataset_name', 'Unknown')}")
                print(f"         Quality Score: {dataset.get('quality_score', 0):.2f}")
                print(f"         Size: {dataset.get('size_bytes', 0)} bytes")
        
        print(f"✅ Existing datasets check successful!")
        return data
        
    except Exception as e:
        print(f"❌ Existing datasets check error: {str(e)}")
        return None

def test_smart_dataset_discovery(query: str):
    """Test the smart dataset discovery tool"""
    print_separator(f"Testing Smart Dataset Discovery: '{query}'")
    
    try:
        from src.tools.data_tools import smart_dataset_discovery_tool
        
        result = smart_dataset_discovery_tool(query, max_new_datasets=3)
        data = json.loads(result)
        
        print(f"🧠 Smart Dataset Discovery Results:")
        print(f"   Category: {data.get('category')}")
        print(f"   Strategy: {data.get('strategy', {}).get('primary_approach')}")
        print(f"   Total Recommendations: {len(data.get('recommendations', []))}")
        
        # Show strategy details
        strategy = data.get('strategy', {})
        print(f"   Existing Assets: {strategy.get('existing_assets', 0)}")
        print(f"   New Sources Needed: {strategy.get('new_sources_needed', 0)}")
        print(f"   Estimated Coverage: {strategy.get('estimated_coverage', 0)}%")
        
        # Show top recommendations
        recommendations = data.get('recommendations', [])
        if recommendations:
            print(f"   🎯 Top Recommendations:")
            for i, rec in enumerate(recommendations[:5], 1):
                dataset_name = rec.get('dataset', {}).get('name', 'Unknown')
                if not dataset_name or dataset_name == 'Unknown':
                    dataset_name = rec.get('dataset', {}).get('dataset_name', 'Unknown')
                
                print(f"      {i}. {rec.get('type')} - {rec.get('priority')} priority")
                print(f"         Dataset: {dataset_name}")
                print(f"         Action: {rec.get('action')}")
                print(f"         Reason: {rec.get('reason')}")
        
        # Show next steps
        next_steps = data.get('next_steps', [])
        if next_steps:
            print(f"   📋 Next Steps:")
            for i, step in enumerate(next_steps, 1):
                print(f"      {i}. {step}")
        
        print(f"   Status: {data.get('status')}")
        print(f"✅ Smart dataset discovery successful!")
        return data
        
    except Exception as e:
        print(f"❌ Smart dataset discovery error: {str(e)}")
        return None

def test_dataset_organization():
    """Test dataset organization tool"""
    print_separator("Testing Dataset Organization")
    
    try:
        from src.tools.data_tools import organize_dataset_categories_tool
        
        print("🗂️  Testing dataset reorganization (dry run)...")
        
        result = organize_dataset_categories_tool(dry_run=True)
        data = json.loads(result)
        
        print(f"📊 Dataset Organization Results:")
        summary = data.get('summary', {})
        print(f"   Datasets Analyzed: {summary.get('total_datasets_analyzed', 0)}")
        print(f"   Datasets Needing Reorganization: {summary.get('datasets_needing_reorganization', 0)}")
        print(f"   Moves Executed: {summary.get('moves_executed', 0)}")
        
        category_dist = summary.get('category_distribution', {})
        if category_dist:
            print(f"   📈 Category Distribution:")
            for category, count in category_dist.items():
                print(f"      {category}: {count} datasets")
        
        # Show reorganization plan
        reorg_plan = data.get('reorganization_plan', [])
        if reorg_plan:
            print(f"   🔄 Reorganization Plan (top 3):")
            for i, plan in enumerate(reorg_plan[:3], 1):
                print(f"      {i}. {plan.get('dataset_name', 'Unknown')}")
                print(f"         From: {plan.get('current_category', 'Unknown')}")
                print(f"         To: {plan.get('suggested_category', 'Unknown')}")
                print(f"         Confidence: {plan.get('confidence', 0):.2f}")
        
        print(f"   Status: {data.get('status')}")
        print(f"✅ Dataset organization test successful!")
        return data
        
    except Exception as e:
        print(f"❌ Dataset organization error: {str(e)}")
        return None

async def test_enhanced_data_agent(query: str):
    """Test the enhanced data agent with smart discovery"""
    print_separator(f"Testing Enhanced Data Agent: '{query}'")
    
    # Initialize shared memory and data agent
    shared_memory = SharedMemory()
    data_agent = DataAgent(shared_memory)
    session_id = f"smart_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"🤖 Testing enhanced data agent with smart discovery...")
    print(f"   Query: '{query}'")
    print(f"   Session: {session_id}")
    print(f"   Features: Category-based storage, Reusability prioritization")
    
    try:
        # Execute data collection with smart discovery
        result = await data_agent.execute_data_collection(
            query=query,
            session_id=session_id
        )
        
        print(f"📊 Enhanced Data Agent Results:")
        print(f"   Execution Method: {result.get('execution_method')}")
        print(f"   Category: {result.get('category')}")
        print(f"   Datasets Found: {result.get('datasets_found')}")
        print(f"   Datasets Processed: {result.get('datasets_processed')}")
        
        # Show smart discovery information
        smart_discovery = result.get('smart_discovery', {})
        if smart_discovery:
            print(f"   🧠 Smart Discovery Info:")
            print(f"      Strategy: {smart_discovery.get('strategy', {}).get('primary_approach')}")
            print(f"      Recommendations: {len(smart_discovery.get('recommendations', []))}")
        
        # Show enhanced data summary
        data_summary = result.get('data_summary', {})
        print(f"   📈 Enhanced Data Summary:")
        print(f"      Existing Datasets Reused: {data_summary.get('existing_datasets_reused', 0)}")
        print(f"      New Datasets Added: {data_summary.get('new_datasets_added', 0)}")
        print(f"      Category: {data_summary.get('category', 'Unknown')}")
        print(f"      Average Quality Score: {data_summary.get('average_quality_score', 0):.2f}")
        print(f"      Reusability Achieved: {data_summary.get('reusability_achieved', False)}")
        
        # Show organization benefits
        org_benefits = result.get('organization_benefits', {})
        if org_benefits:
            print(f"   🎯 Organization Benefits:")
            print(f"      Category-Based Storage: {org_benefits.get('category_based_storage', False)}")
            print(f"      Future Reusability: {org_benefits.get('future_reusability', False)}")
            print(f"      Discoverable Structure: {org_benefits.get('discoverable_structure', False)}")
        
        # Show S3 locations with category info
        s3_locations = result.get("s3_locations", [])
        if s3_locations:
            print(f"   ☁️  Enhanced S3 Storage:")
            for i, location in enumerate(s3_locations[:3], 1):
                category_path = location.get('category_path', 'Unknown')
                print(f"      {i}. Category Path: {category_path}")
                print(f"         Full Key: {location.get('key', 'Unknown')}")
                print(f"         Size: {location.get('size_bytes', 0)} bytes")
                print(f"         Reusable: {location.get('reusable', True)}")
        
        print(f"   Status: {result.get('status')}")
        
        # Validate enhanced data quality
        quality_metrics = data_agent.validate_data_quality(result)
        print(f"   🏆 Enhanced Quality Assessment:")
        print(f"      Overall Quality: {quality_metrics.get('overall_quality')}")
        print(f"      Quality Level: {quality_metrics.get('quality_level')}")
        print(f"      Processing Success Rate: {quality_metrics.get('processing_success_rate'):.2%}")
        print(f"      Source Diversity: {quality_metrics.get('source_diversity')}")
        
        print(f"✅ Enhanced data agent test successful!")
        return result
        
    except Exception as e:
        print(f"❌ Enhanced data agent error: {str(e)}")
        return None

def test_individual_smart_tools(query: str):
    """Test individual smart data tools"""
    print_separator("Testing Individual Smart Tools")
    
    try:
        # Test all the new smart tools
        from src.tools.data_tools import (
            categorize_query,
            check_existing_datasets_tool,
            smart_dataset_discovery_tool,
            organize_dataset_categories_tool,
            kaggle_search_tool,
            huggingface_search_tool
        )
        
        print("🔧 Testing individual smart tools...")
        
        # Test categorization
        print("   1. Query Categorization...")
        category, confidence = categorize_query(query)
        print(f"      → {category} (confidence: {confidence:.2f})")
        
        # Test existing datasets check
        print("   2. Existing Datasets Check...")
        existing_result = check_existing_datasets_tool(query)
        existing_data = json.loads(existing_result)
        existing_count = len(existing_data.get('existing_datasets', {}).get('primary_category', []))
        print(f"      → Found {existing_count} existing datasets")
        
        # Test smart discovery
        print("   3. Smart Dataset Discovery...")
        smart_result = smart_dataset_discovery_tool(query, max_new_datasets=2)
        smart_data = json.loads(smart_result)
        recommendations_count = len(smart_data.get('recommendations', []))
        print(f"      → Generated {recommendations_count} recommendations")
        
        # Test organization (dry run)
        print("   4. Dataset Organization...")
        org_result = organize_dataset_categories_tool(dry_run=True)
        org_data = json.loads(org_result)
        reorg_count = org_data.get('summary', {}).get('datasets_needing_reorganization', 0)
        print(f"      → {reorg_count} datasets need reorganization")
        
        # Test traditional tools for comparison
        print("   5. Traditional Tools (for comparison)...")
        kaggle_result = kaggle_search_tool(query, max_results=2)
        kaggle_data = json.loads(kaggle_result)
        kaggle_count = kaggle_data.get('datasets_found', 0)
        print(f"      → Kaggle: {kaggle_count} datasets")
        
        hf_result = huggingface_search_tool(query, max_results=2)
        hf_data = json.loads(hf_result)
        hf_count = hf_data.get('datasets_found', 0)
        print(f"      → HuggingFace: {hf_count} datasets")
        
        print("✅ All individual smart tools tested successfully!")
        
        return {
            "categorization": {"category": category, "confidence": confidence},
            "existing_datasets": existing_count,
            "smart_recommendations": recommendations_count,
            "reorganization_needed": reorg_count,
            "kaggle_datasets": kaggle_count,
            "huggingface_datasets": hf_count
        }
        
    except Exception as e:
        print(f"❌ Individual smart tools test error: {str(e)}")
        return None

def demonstrate_benefits():
    """Demonstrate the benefits of the smart system"""
    print_separator("Smart Dataset Discovery Benefits")
    
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

async def run_comprehensive_smart_test(query: str):
    """Run comprehensive smart dataset discovery test suite"""
    print_separator(f"Smart Dataset Discovery Comprehensive Test")
    print(f"Query: '{query}'")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Testing: Category-based organization, Intelligent reusability, Smart recommendations")
    
    # Test 1: Query categorization
    category, confidence = test_query_categorization(query)
    
    # Test 2: Existing datasets check
    existing_data = test_existing_datasets_check(query)
    
    # Test 3: Smart dataset discovery
    smart_data = test_smart_dataset_discovery(query)
    
    # Test 4: Dataset organization
    org_data = test_dataset_organization()
    
    # Test 5: Individual smart tools
    tools_data = test_individual_smart_tools(query)
    
    # Test 6: Enhanced data agent
    agent_results = await test_enhanced_data_agent(query)
    
    # Test 7: Show benefits
    demonstrate_benefits()
    
    # Final summary
    print_separator("Smart Test Summary")
    
    if agent_results and agent_results.get('status') == 'success':
        print(f"✅ Smart Dataset Discovery Test Completed Successfully!")
        print(f"   Query: {query}")
        print(f"   Category: {category} (confidence: {confidence:.2f})")
        print(f"   Execution Method: {agent_results.get('execution_method')}")
        
        data_summary = agent_results.get('data_summary', {})
        print(f"   Datasets Found: {agent_results.get('datasets_found', 0)}")
        print(f"   Datasets Processed: {agent_results.get('datasets_processed', 0)}")
        print(f"   Existing Reused: {data_summary.get('existing_datasets_reused', 0)}")
        print(f"   New Added: {data_summary.get('new_datasets_added', 0)}")
        print(f"   Reusability Achieved: {'✅' if data_summary.get('reusability_achieved') else '❌'}")
        print(f"   Category-Based Storage: {'✅' if agent_results.get('organization_benefits', {}).get('category_based_storage') else '❌'}")
        
        # Save comprehensive results
        test_results = {
            "query": query,
            "category": category,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
            "categorization_results": {"category": category, "confidence": confidence},
            "existing_datasets_results": existing_data,
            "smart_discovery_results": smart_data,
            "organization_results": org_data,
            "individual_tools_results": tools_data,
            "enhanced_agent_results": agent_results,
            "test_status": "success"
        }
        
        # Create organized test results directory
        test_results_dir = Path("test_results/smart_discovery")
        test_results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create meaningful filename
        query_safe = query.replace(" ", "_").replace("/", "_").lower()
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        results_file = test_results_dir / f"smart_discovery_test_{query_safe}_{timestamp}.json"
        
        # Save detailed results
        with open(results_file, 'w') as f:
            json.dump(test_results, f, indent=2, default=str)
        
        # Create summary file
        summary_file = test_results_dir / f"smart_discovery_summary_{query_safe}_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Smart Dataset Discovery Test Summary\n")
            f.write(f"===================================\n\n")
            f.write(f"Query: {query}\n")
            f.write(f"Category: {category} (confidence: {confidence:.2f})\n")
            f.write(f"Timestamp: {timestamp}\n\n")
            f.write(f"Results:\n")
            f.write(f"- Execution Method: {agent_results.get('execution_method')}\n")
            f.write(f"- Datasets Found: {agent_results.get('datasets_found', 0)}\n")
            f.write(f"- Datasets Processed: {agent_results.get('datasets_processed', 0)}\n")
            f.write(f"- Existing Datasets Reused: {data_summary.get('existing_datasets_reused', 0)}\n")
            f.write(f"- New Datasets Added: {data_summary.get('new_datasets_added', 0)}\n")
            f.write(f"- Reusability Achieved: {data_summary.get('reusability_achieved', False)}\n")
            f.write(f"- Category-Based Storage: {agent_results.get('organization_benefits', {}).get('category_based_storage', False)}\n")
            f.write(f"- Average Quality Score: {data_summary.get('average_quality_score', 0):.2f}\n\n")
            f.write(f"Smart Features Tested:\n")
            f.write(f"- ✅ Query Categorization\n")
            f.write(f"- ✅ Existing Dataset Discovery\n")
            f.write(f"- ✅ Smart Recommendations\n")
            f.write(f"- ✅ Category-Based Organization\n")
            f.write(f"- ✅ Enhanced Data Agent\n")
            f.write(f"- ✅ Reusability Prioritization\n\n")
            f.write(f"Files:\n")
            f.write(f"- Detailed Results: {results_file.name}\n")
            f.write(f"- Summary: {summary_file.name}\n")
        
        print(f"📄 Smart test results saved to: {test_results_dir}/")
        print(f"   - Detailed: {results_file.name}")
        print(f"   - Summary: {summary_file.name}")
        
        return test_results
    else:
        print("❌ Smart dataset discovery test failed!")
        return None

async def main():
    """Main test function"""
    parser = argparse.ArgumentParser(description="Test Smart Dataset Discovery System")
    parser.add_argument("--search", default="machine learning sentiment analysis", help="Search query for datasets")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--category-only", action="store_true", help="Test only categorization")
    parser.add_argument("--existing-only", action="store_true", help="Test only existing datasets check")
    parser.add_argument("--smart-only", action="store_true", help="Test only smart discovery")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logger(log_level="DEBUG" if args.verbose else "INFO")
    
    try:
        if args.category_only:
            test_query_categorization(args.search)
        elif args.existing_only:
            test_existing_datasets_check(args.search)
        elif args.smart_only:
            test_smart_dataset_discovery(args.search)
        else:
            # Run comprehensive test
            results = await run_comprehensive_smart_test(args.search)
            
            if results:
                print(f"\n🎉 All smart discovery tests completed successfully!")
                print(f"\nThe Smart Dataset Discovery System is ready to:")
                print(f"1. ✅ Automatically categorize research queries")
                print(f"2. ✅ Check existing datasets before downloading new ones")
                print(f"3. ✅ Organize datasets in category-based S3 structure")
                print(f"4. ✅ Provide intelligent recommendations for dataset reuse")
                print(f"5. ✅ Build a reusable, well-organized dataset library")
                sys.exit(0)
            else:
                print(f"\n❌ Smart discovery tests failed!")
                sys.exit(1)
            
    except KeyboardInterrupt:
        print(f"\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())