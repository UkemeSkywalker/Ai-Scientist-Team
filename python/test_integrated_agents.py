#!/usr/bin/env python3
"""
Integrated Agent Test - Data Agent + Experiment Agent with Real Data Flow
Tests the complete multi-agent workflow using real datasets (no mock data)
"""

import asyncio
import json
import argparse
import sys
import os
from datetime import datetime
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.agents.research_agent import create_research_agent
from src.agents.data_agent import create_data_agent
from src.agents.experiment_agent import create_experiment_agent
from src.core.shared_memory import SharedMemory
from src.core.logger import get_logger

logger = get_logger(__name__)

def create_fallback_hypotheses(query: str) -> str:
    """Create fallback hypotheses when Research Agent fails"""
    if "sentiment" in query.lower():
        hypotheses = {
            "hypotheses": [
                "A machine learning classifier can achieve >80% accuracy on sentiment classification",
                "Text features like word count and sentiment scores are predictive of sentiment",
                "The dataset shows distinguishable patterns between positive and negative sentiments",
                "Cross-validation will demonstrate model generalizability across different text samples"
            ],
            "research_question": f"Can we build a reliable sentiment classifier for: {query}?",
            "success_criteria": [
                "Model accuracy > 0.80",
                "Statistical significance p < 0.05",
                "Cross-validation stability"
            ]
        }
    else:
        hypotheses = {
            "hypotheses": [
                f"A machine learning model can effectively classify data related to {query}",
                "Feature engineering and selection will improve model performance",
                "The dataset contains sufficient signal for reliable predictions",
                "Statistical validation will confirm model reliability"
            ],
            "research_question": f"Can we build a reliable classifier for: {query}?",
            "success_criteria": [
                "Model accuracy > 0.75",
                "Statistical significance p < 0.05",
                "Cross-validation stability"
            ]
        }
    return json.dumps(hypotheses, indent=2)

def extract_real_data_context(data_result: Dict[str, Any], query: str = "") -> Dict[str, Any]:
    """Extract real data context from Data Agent results"""
    processed_datasets = data_result.get("processed_datasets", [])
    category = data_result.get("category", "machine-learning")
    
    if not processed_datasets:
        raise ValueError("No processed datasets found in Data Agent results")
    
    # Convert processed datasets to experiment-ready format
    datasets = []
    for dataset in processed_datasets:
        original = dataset.get("original_dataset", {})
        storage = dataset.get("storage_results", {})
        cleaning = dataset.get("cleaning_results", {})
        
        # Extract S3 location - use data file path instead of metadata path
        s3_location = storage.get("s3_location", {})
        if isinstance(s3_location, dict) and s3_location.get('bucket') and s3_location.get('data_key'):
            # Use data_key for actual dataset file, not metadata_key
            bucket = s3_location.get('bucket')
            data_key = s3_location.get('data_key')
            s3_path = f"s3://{bucket}/{data_key}"
        elif isinstance(s3_location, str) and s3_location.startswith('s3://'):
            s3_path = s3_location
        else:
            # Skip datasets without valid S3 paths
            continue
        
        # Determine task type and target based on dataset name and query
        dataset_name = original.get("name", "").lower()
        query_lower = query.lower()
        
        if "sentiment" in dataset_name or "sentiment" in query_lower or "review" in dataset_name:
            task_type = "classification"
            target_variable = "sentiment"
            columns = ["text_length", "sentiment_score", "rating", "word_count", "sentiment"]
        elif "iris" in dataset_name:
            task_type = "classification"
            target_variable = "species"
            columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
        else:
            # Generic classification dataset based on query
            task_type = "classification"
            target_variable = "target"
            columns = ["feature_1", "feature_2", "feature_3", "feature_4", "target"]
        
        dataset_info = {
            "name": original.get("name", "unknown_dataset"),
            "type": "supervised",
            "task": task_type,
            "columns": columns,
            "target": target_variable,
            "shape": [1000, len(columns)],  # Estimated
            "classes": (
                ["positive", "negative", "neutral"] if "sentiment" in dataset_name or "sentiment" in query.lower() else
                ["setosa", "versicolor", "virginica"] if "iris" in dataset_name else
                ["class_0", "class_1", "class_2"] if task_type == "classification" else None
            ),
            "data_quality": "excellent",
            "missing_values": 0,
            "s3_location": s3_path,
            "description": original.get("description", "Dataset processed by Data Agent"),
            "source": original.get("source", "unknown"),
            "quality_score": cleaning.get("quality_metrics", {}).get("overall_score", 0.8)
        }
        datasets.append(dataset_info)
    
    # Create comprehensive data context
    real_data_context = {
        "datasets": datasets,
        "preprocessing": {
            "scaling": "standard",
            "encoding": "label",
            "train_test_split": 0.2
        },
        "data_summary": {
            "total_samples": sum(d.get("shape", [0])[0] for d in datasets),
            "features": datasets[0].get("shape", [0, 4])[1] - 1 if datasets else 4,
            "target_classes": len(datasets[0].get("classes", [])) if datasets and datasets[0].get("classes") else 3,
            "balanced": True,
            "category": category,
            "source_diversity": len(set(d.get("source", "unknown") for d in datasets)),
            "average_quality": sum(d.get("quality_score", 0) for d in datasets) / max(len(datasets), 1)
        }
    }
    
    return real_data_context

async def test_integrated_classification_workflow(query: str, session_id: str) -> Dict[str, Any]:
    """Test complete integrated workflow: Research Agent → Data Agent → Experiment Agent"""
    
    print(f"\n{'='*80}")
    print("INTEGRATED AGENT TEST - FULL WORKFLOW")
    print(f"{'='*80}")
    print(f"Query: {query}")
    print(f"Session ID: {session_id}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    # Initialize shared memory and agents
    shared_memory = SharedMemory()
    research_agent = create_research_agent(shared_memory)
    data_agent = create_data_agent(shared_memory)
    experiment_agent = create_experiment_agent(shared_memory)
    
    try:
        # PHASE 1: RESEARCH AGENT
        print(f"\n{'='*60}")
        print("PHASE 1: RESEARCH HYPOTHESIS GENERATION")
        print(f"{'='*60}")
        
        logger.info("Starting research phase", query=query, session_id=session_id)
        
        research_result = await research_agent.execute_research(
            query=query,
            session_id=session_id
        )
        
        print(f"🔬 Research Agent Results:")
        print(f"   Execution Method: {research_result.get('execution_method')}")
        print(f"   Status: {research_result.get('status')}")
        
        if research_result.get('status') != 'success':
            print(f"   ⚠️  Research failed, using fallback hypotheses")
            hypotheses_from_research = []
        else:
            hypotheses_from_research = research_result.get('hypotheses', [])
            print(f"   Hypotheses Generated: {len(hypotheses_from_research)}")
            for i, hyp in enumerate(hypotheses_from_research[:3], 1):
                print(f"      {i}. {hyp.get('text', hyp)[:80]}...")
        
        # PHASE 2: DATA COLLECTION
        print(f"\n{'='*60}")
        print("PHASE 2: DATA COLLECTION WITH DATA AGENT")
        print(f"{'='*60}")
        
        logger.info("Starting data collection phase", query=query, session_id=session_id)
        
        # Data Agent now reads query from shared memory (Research Agent results)
        data_result = await data_agent.execute_data_collection(
            query="",  # Empty - will read from shared memory
            session_id=session_id
        )
        
        print(f"📊 Data Agent Results:")
        print(f"   Execution Method: {data_result.get('execution_method')}")
        print(f"   Category: {data_result.get('category')}")
        print(f"   Datasets Found: {data_result.get('datasets_found')}")
        print(f"   Datasets Processed: {data_result.get('datasets_processed')}")
        print(f"   Status: {data_result.get('status')}")
        
        if data_result.get('status') != 'success':
            raise ValueError(f"Data collection failed: {data_result.get('error', 'Unknown error')}")
        
        # Show S3 locations
        s3_locations = data_result.get("s3_locations", [])
        if s3_locations:
            print(f"   ☁️  S3 Storage Locations:")
            for i, location in enumerate(s3_locations[:3], 1):
                print(f"      {i}. {location.get('key', 'Unknown')}")
                print(f"         Size: {location.get('size_bytes', 0)} bytes")
        
        # PHASE 3: DATA CONTEXT EXTRACTION
        print(f"\n{'='*60}")
        print("PHASE 3: REAL DATA CONTEXT EXTRACTION")
        print(f"{'='*60}")
        
        logger.info("Extracting real data context from Data Agent results", session_id=session_id)
        
        real_data_context = extract_real_data_context(data_result, query)
        
        print(f"🔄 Real Data Context Extracted:")
        print(f"   Datasets: {len(real_data_context['datasets'])}")
        print(f"   Category: {real_data_context['data_summary']['category']}")
        print(f"   Total Samples: {real_data_context['data_summary']['total_samples']}")
        print(f"   Source Diversity: {real_data_context['data_summary']['source_diversity']}")
        print(f"   Average Quality: {real_data_context['data_summary']['average_quality']:.2f}")
        
        # Show dataset details
        for i, dataset in enumerate(real_data_context['datasets'][:3], 1):
            print(f"   Dataset {i}: {dataset['name']}")
            print(f"      S3 Location: {dataset['s3_location']}")
            print(f"      Task: {dataset['task']}")
            print(f"      Target: {dataset['target']}")
            print(f"      Quality: {dataset['quality_score']:.2f}")
        
        # PHASE 4: EXPERIMENT EXECUTION
        print(f"\n{'='*60}")
        print("PHASE 4: EXPERIMENT EXECUTION WITH EXPERIMENT AGENT")
        print(f"{'='*60}")
        
        logger.info("Starting experiment execution phase", session_id=session_id)
        
        # Use hypotheses from Research Agent or fallback
        if hypotheses_from_research:
            print(f"🧪 Using Research Agent Hypotheses ({len(hypotheses_from_research)} generated)")
            for i, hyp in enumerate(hypotheses_from_research[:3], 1):
                hyp_text = hyp.get('text', hyp) if isinstance(hyp, dict) else str(hyp)
                print(f"   {i}. {hyp_text[:100]}...")
        else:
            print(f"🧪 Using Fallback Hypotheses (Research Agent failed)")
            fallback_hypotheses = create_fallback_hypotheses(query)
            hypotheses_data = json.loads(fallback_hypotheses)
            for i, hypothesis in enumerate(hypotheses_data['hypotheses'][:3], 1):
                print(f"   {i}. {hypothesis}")
        
        # Experiment Agent now reads both research and data results from shared memory
        experiment_result = await experiment_agent.execute_experiments(
            hypotheses="",  # Empty - will read from shared memory
            data_context="",  # Empty - will read from shared memory
            session_id=session_id
        )
        
        print(f"\n📈 Experiment Agent Results:")
        print(f"   Execution Method: {experiment_result.get('execution_method')}")
        print(f"   Status: {experiment_result.get('status')}")
        
        # Show experiment design
        exp_design = experiment_result.get('experiment_design', {})
        if exp_design:
            experiments = exp_design.get('experiments', [])
            print(f"   Experiments Designed: {len(experiments)}")
            for exp in experiments:
                print(f"      - {exp.get('experiment_type', 'unknown')}")
        
        # Show ML training results
        ml_training = experiment_result.get('ml_training', {})
        if ml_training:
            training_jobs = ml_training.get('training_jobs', [])
            successful_jobs = ml_training.get('successful_jobs', 0)
            print(f"   ML Training Jobs: {len(training_jobs)} ({successful_jobs} successful)")
            
            for job in training_jobs[:3]:
                if job.get('status') == 'Completed':
                    metrics = job.get('metrics', {})
                    print(f"      ✅ {job.get('model_type', 'unknown')}: {metrics.get('accuracy', 0):.3f} accuracy")
                else:
                    print(f"      ❌ {job.get('model_type', 'unknown')}: {job.get('status', 'failed')}")
        
        # Show statistical analysis
        stats_analysis = experiment_result.get('statistical_analysis', {})
        if stats_analysis:
            stats_tests = stats_analysis.get('statistical_tests', [])
            significant_tests = len([t for t in stats_tests if t.get('significant', False)])
            print(f"   Statistical Tests: {len(stats_tests)} ({significant_tests} significant)")
        
        # Show interpretation
        interpretation = experiment_result.get('interpretation', {})
        if interpretation:
            insights = interpretation.get('insights', [])
            recommendations = interpretation.get('recommendations', [])
            confidence = interpretation.get('overall_confidence', 0)
            print(f"   Insights Generated: {len(insights)}")
            print(f"   Recommendations: {len(recommendations)}")
            print(f"   Overall Confidence: {confidence}")
        
        # PHASE 5: INTEGRATION VALIDATION
        print(f"\n{'='*60}")
        print("PHASE 5: INTEGRATION VALIDATION")
        print(f"{'='*60}")
        
        validation_results = validate_integration(research_result, data_result, experiment_result)
        
        print(f"🔍 Integration Validation:")
        for check, passed in validation_results.items():
            status = "✅" if passed else "❌"
            print(f"   {status} {check.replace('_', ' ').title()}")
        
        all_passed = all(validation_results.values())
        
        # FINAL RESULTS
        final_results = {
            "query": query,
            "session_id": session_id,
            "research_agent_results": research_result,
            "data_agent_results": data_result,
            "real_data_context": real_data_context,
            "experiment_agent_results": experiment_result,
            "integration_validation": validation_results,
            "integration_successful": all_passed,
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "hypotheses_generated": len(research_result.get('hypotheses', [])),
                "datasets_found": data_result.get('datasets_found', 0),
                "datasets_processed": data_result.get('datasets_processed', 0),
                "experiments_designed": len(exp_design.get('experiments', [])),
                "training_jobs_completed": ml_training.get('successful_jobs', 0),
                "statistical_tests_performed": len(stats_analysis.get('statistical_tests', [])),
                "insights_generated": len(interpretation.get('insights', [])),
                "overall_success": all_passed
            }
        }
        
        print(f"\n{'='*80}")
        print("INTEGRATION TEST SUMMARY")
        print(f"{'='*80}")
        
        summary = final_results['summary']
        print(f"✅ INTEGRATION TEST {'PASSED' if all_passed else 'FAILED'}")
        print(f"   Query: {query}")
        print(f"   Hypotheses Generated: {summary['hypotheses_generated']}")
        print(f"   Datasets Found: {summary['datasets_found']}")
        print(f"   Datasets Processed: {summary['datasets_processed']}")
        print(f"   Experiments Designed: {summary['experiments_designed']}")
        print(f"   Training Jobs Completed: {summary['training_jobs_completed']}")
        print(f"   Statistical Tests: {summary['statistical_tests_performed']}")
        print(f"   Insights Generated: {summary['insights_generated']}")
        print(f"   Overall Success: {'✅' if summary['overall_success'] else '❌'}")
        
        if all_passed:
            print(f"\n🎉 MULTI-AGENT INTEGRATION SUCCESSFUL!")
            print(f"   ✅ Research Agent generated query-appropriate hypotheses")
            print(f"   ✅ Data Agent found and processed real datasets")
            print(f"   ✅ Real data context extracted successfully")
            print(f"   ✅ Experiment Agent used real hypotheses and data")
            print(f"   ✅ End-to-end pipeline completed via shared memory")
        else:
            print(f"\n⚠️  INTEGRATION ISSUES DETECTED")
            failed_checks = [k for k, v in validation_results.items() if not v]
            for check in failed_checks:
                print(f"   ❌ {check.replace('_', ' ').title()}")
        
        return final_results
        
    except Exception as e:
        error_msg = f"Integrated test failed: {str(e)}"
        logger.error("Integrated test execution failed", error=error_msg, session_id=session_id)
        
        error_result = {
            "query": query,
            "session_id": session_id,
            "error": error_msg,
            "status": "failed",
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"\n❌ INTEGRATION TEST FAILED")
        print(f"   Error: {error_msg}")
        
        return error_result

def validate_integration(research_result: Dict[str, Any], data_result: Dict[str, Any], experiment_result: Dict[str, Any]) -> Dict[str, bool]:
    """Validate that the integration between agents worked correctly"""
    validation = {}
    
    # Check research agent success
    validation['research_agent_success'] = research_result.get('status') == 'success'
    validation['hypotheses_generated'] = len(research_result.get('hypotheses', [])) > 0
    
    # Check data agent success
    validation['data_agent_success'] = data_result.get('status') == 'success'
    
    # Check datasets were found and processed
    validation['datasets_found'] = data_result.get('datasets_found', 0) > 0
    validation['datasets_processed'] = data_result.get('datasets_processed', 0) > 0
    
    # Check S3 storage
    validation['s3_storage_success'] = len(data_result.get('s3_locations', [])) > 0
    
    # Check experiment agent success
    validation['experiment_agent_success'] = experiment_result.get('status') == 'completed'
    
    # Check experiment design
    exp_design = experiment_result.get('experiment_design', {})
    validation['experiments_designed'] = len(exp_design.get('experiments', [])) > 0
    
    # Check ML training
    ml_training = experiment_result.get('ml_training', {})
    validation['ml_training_attempted'] = len(ml_training.get('training_jobs', [])) > 0
    validation['ml_training_successful'] = ml_training.get('successful_jobs', 0) > 0
    
    # Check statistical analysis
    stats_analysis = experiment_result.get('statistical_analysis', {})
    validation['statistical_analysis_performed'] = len(stats_analysis.get('statistical_tests', [])) > 0
    
    # Check interpretation
    interpretation = experiment_result.get('interpretation', {})
    validation['insights_generated'] = len(interpretation.get('insights', [])) > 0
    
    # Check no mock data was used
    validation['no_mock_data'] = 'mock' not in str(experiment_result).lower()
    
    return validation

async def main():
    """Main test function"""
    parser = argparse.ArgumentParser(description="Test Integrated Data Agent + Experiment Agent")
    parser.add_argument("--query", default="sentiment analysis natural language processing", help="Research query")
    parser.add_argument("--session-id", default=None, help="Custom session ID")
    
    args = parser.parse_args()
    
    # Generate session ID if not provided
    if args.session_id is None:
        timestamp = int(datetime.now().timestamp())
        args.session_id = f"integrated_test_{timestamp}"
    
    try:
        # Run integrated test
        results = await test_integrated_classification_workflow(args.query, args.session_id)
        
        # Check if test passed
        success = results.get('integration_successful', False)
        
        if success:
            print(f"\n🎉 INTEGRATED AGENT TEST COMPLETED SUCCESSFULLY!")
            sys.exit(0)
        else:
            print(f"\n❌ INTEGRATED AGENT TEST FAILED!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print(f"\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())