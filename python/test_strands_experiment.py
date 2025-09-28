#!/usr/bin/env python3
"""
Test script for Strands Experiment Agent with ML tools and SageMaker integration.
This script tests the complete experiment workflow including classification tasks.
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

from src.agents.experiment_agent import create_experiment_agent
from src.core.shared_memory import SharedMemory
from src.core.logger import get_logger

logger = get_logger(__name__)


def create_classification_hypotheses() -> str:
    """Create sample hypotheses for classification experiment"""
    hypotheses = {
        "hypotheses": [
            "A Random Forest classifier can achieve >85% accuracy on the iris dataset",
            "Feature importance analysis will show petal length as the most predictive feature",
            "The dataset shows clear separability between species classes",
            "Cross-validation will demonstrate model stability with <5% variance in accuracy"
        ],
        "research_question": "Can we build a reliable classifier for iris species prediction?",
        "success_criteria": [
            "Model accuracy > 0.85",
            "Statistical significance p < 0.05",
            "Cross-validation stability"
        ]
    }
    return json.dumps(hypotheses, indent=2)


def create_regression_hypotheses() -> str:
    """Create sample hypotheses for regression experiment"""
    hypotheses = {
        "hypotheses": [
            "Linear regression can predict house prices with R² > 0.7",
            "Feature correlation analysis will identify key price drivers",
            "Residual analysis will show homoscedasticity",
            "Model performance is consistent across different price ranges"
        ],
        "research_question": "What factors most strongly predict house prices?",
        "success_criteria": [
            "R² score > 0.7",
            "RMSE < 20% of mean price",
            "No significant residual patterns"
        ]
    }
    return json.dumps(hypotheses, indent=2)


def create_classification_data_context() -> str:
    """Create sample data context for classification"""
    data_context = {
        "datasets": [
            {
                "name": "iris_classification",
                "type": "supervised",
                "task": "classification",
                "columns": ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"],
                "target": "species",
                "shape": [150, 5],
                "classes": ["setosa", "versicolor", "virginica"],
                "data_quality": "excellent",
                "missing_values": 0,
                "s3_location": "s3://ai-scientist-data/iris/iris.csv",
                "description": "Classic iris flower classification dataset"
            }
        ],
        "preprocessing": {
            "scaling": "standard",
            "encoding": "label",
            "train_test_split": 0.2
        },
        "data_summary": {
            "total_samples": 150,
            "features": 4,
            "target_classes": 3,
            "balanced": True
        }
    }
    return json.dumps(data_context, indent=2)


def create_regression_data_context() -> str:
    """Create sample data context for regression"""
    data_context = {
        "datasets": [
            {
                "name": "house_prices",
                "type": "supervised",
                "task": "regression",
                "columns": ["bedrooms", "bathrooms", "sqft", "age", "location_score", "price"],
                "target": "price",
                "shape": [1000, 6],
                "data_quality": "good",
                "missing_values": 5,
                "s3_location": "s3://ai-scientist-data/housing/prices.csv",
                "description": "House price prediction dataset"
            }
        ],
        "preprocessing": {
            "scaling": "standard",
            "missing_value_strategy": "median",
            "train_test_split": 0.2
        },
        "data_summary": {
            "total_samples": 1000,
            "features": 5,
            "price_range": [100000, 800000],
            "outliers_detected": 12
        }
    }
    return json.dumps(data_context, indent=2)


async def test_classification_experiment(agent, session_id: str) -> Dict[str, Any]:
    """Test classification experiment workflow"""
    logger.info("Starting classification experiment test", session_id=session_id)
    
    hypotheses = create_classification_hypotheses()
    data_context = create_classification_data_context()
    
    print(f"\n{'='*60}")
    print("CLASSIFICATION EXPERIMENT TEST")
    print(f"{'='*60}")
    print(f"Session ID: {session_id}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    print(f"\nHypotheses:")
    print(hypotheses)
    
    print(f"\nData Context:")
    print(data_context)
    
    print(f"\n{'='*60}")
    print("EXECUTING EXPERIMENT WORKFLOW...")
    print(f"{'='*60}")
    
    # Execute the experiment
    results = await agent.execute_experiments(hypotheses, data_context, session_id)
    
    print(f"\n{'='*60}")
    print("EXPERIMENT RESULTS")
    print(f"{'='*60}")
    print(json.dumps(results, indent=2))
    
    return results


async def test_regression_experiment(agent, session_id: str) -> Dict[str, Any]:
    """Test regression experiment workflow"""
    logger.info("Starting regression experiment test", session_id=session_id)
    
    hypotheses = create_regression_hypotheses()
    data_context = create_regression_data_context()
    
    print(f"\n{'='*60}")
    print("REGRESSION EXPERIMENT TEST")
    print(f"{'='*60}")
    print(f"Session ID: {session_id}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    print(f"\nHypotheses:")
    print(hypotheses)
    
    print(f"\nData Context:")
    print(data_context)
    
    print(f"\n{'='*60}")
    print("EXECUTING EXPERIMENT WORKFLOW...")
    print(f"{'='*60}")
    
    # Execute the experiment
    results = await agent.execute_experiments(hypotheses, data_context, session_id)
    
    print(f"\n{'='*60}")
    print("EXPERIMENT RESULTS")
    print(f"{'='*60}")
    print(json.dumps(results, indent=2))
    
    return results


async def test_statistical_analysis(agent, session_id: str) -> Dict[str, Any]:
    """Test statistical analysis capabilities"""
    logger.info("Starting statistical analysis test", session_id=session_id)
    
    hypotheses = {
        "hypotheses": [
            "Variables show normal distribution patterns",
            "Significant correlations exist between features",
            "No multicollinearity issues present"
        ]
    }
    
    data_context = {
        "datasets": [
            {
                "name": "statistical_test_data",
                "type": "exploratory",
                "columns": ["var1", "var2", "var3", "var4", "var5"],
                "shape": [500, 5],
                "data_quality": "good"
            }
        ]
    }
    
    print(f"\n{'='*60}")
    print("STATISTICAL ANALYSIS TEST")
    print(f"{'='*60}")
    
    results = await agent.execute_experiments(
        json.dumps(hypotheses), 
        json.dumps(data_context), 
        session_id
    )
    
    print(json.dumps(results, indent=2))
    return results


def validate_experiment_results(results: Dict[str, Any], experiment_type: str) -> bool:
    """Validate that experiment results meet expected criteria"""
    logger.info(f"Validating {experiment_type} experiment results")
    
    validation_passed = True
    issues = []
    
    # Check basic structure
    required_keys = ["experiment_design", "ml_training", "statistical_analysis", "interpretation"]
    for key in required_keys:
        if key not in results:
            issues.append(f"Missing required key: {key}")
            validation_passed = False
    
    # Validate ML training results
    if "ml_training" in results:
        ml_results = results["ml_training"]
        if "training_jobs" not in ml_results:
            issues.append("Missing training jobs in ML training results")
            validation_passed = False
        else:
            training_jobs = ml_results["training_jobs"]
            if not training_jobs:
                issues.append("No training jobs found")
                validation_passed = False
            else:
                for job in training_jobs:
                    if "metrics" not in job:
                        issues.append(f"Missing metrics in training job {job.get('job_name', 'unknown')}")
                        validation_passed = False
                    else:
                        metrics = job["metrics"]
                        if experiment_type == "classification":
                            if "accuracy" not in metrics:
                                issues.append("Missing accuracy metric for classification")
                                validation_passed = False
                            elif metrics["accuracy"] < 0.5:
                                issues.append(f"Suspiciously low accuracy: {metrics['accuracy']}")
                        elif experiment_type == "regression":
                            if "r2_score" not in metrics:
                                issues.append("Missing R² score for regression")
                                validation_passed = False
    
    # Validate statistical analysis
    if "statistical_analysis" in results:
        stats_results = results["statistical_analysis"]
        if "error" in stats_results:
            issues.append(f"Statistical analysis failed: {stats_results['error']}")
            validation_passed = False
        elif "statistical_tests" not in stats_results:
            issues.append("Missing statistical tests")
            validation_passed = False
        elif len(stats_results["statistical_tests"]) == 0:
            issues.append("No statistical tests performed")
            validation_passed = False
    
    # Validate interpretation
    if "interpretation" in results:
        interpretation = results["interpretation"]
        required_interp_keys = ["insights", "recommendations", "confidence_scores"]
        for key in required_interp_keys:
            if key not in interpretation:
                issues.append(f"Missing interpretation key: {key}")
                validation_passed = False
    
    # Print validation results
    print(f"\n{'='*60}")
    print("VALIDATION RESULTS")
    print(f"{'='*60}")
    
    if validation_passed:
        print("✅ All validation checks passed!")
    else:
        print("❌ Validation issues found:")
        for issue in issues:
            print(f"  - {issue}")
    
    return validation_passed


async def run_comprehensive_test():
    """Run comprehensive test of the Experiment Agent"""
    print(f"\n{'='*80}")
    print("STRANDS EXPERIMENT AGENT COMPREHENSIVE TEST")
    print(f"{'='*80}")
    print(f"Test started at: {datetime.now().isoformat()}")
    
    # Initialize shared memory and agent
    shared_memory = SharedMemory()
    agent = create_experiment_agent(shared_memory)
    
    # Test results storage
    test_results = {}
    
    try:
        # Test 1: Classification Experiment
        session_id_1 = f"classification_test_{int(datetime.now().timestamp())}"
        classification_results = await test_classification_experiment(agent, session_id_1)
        test_results["classification"] = {
            "results": classification_results,
            "validation_passed": validate_experiment_results(classification_results, "classification")
        }
        
        # Test 2: Regression Experiment
        session_id_2 = f"regression_test_{int(datetime.now().timestamp())}"
        regression_results = await test_regression_experiment(agent, session_id_2)
        test_results["regression"] = {
            "results": regression_results,
            "validation_passed": validate_experiment_results(regression_results, "regression")
        }
        
        # Test 3: Statistical Analysis
        session_id_3 = f"statistical_test_{int(datetime.now().timestamp())}"
        statistical_results = await test_statistical_analysis(agent, session_id_3)
        test_results["statistical"] = {
            "results": statistical_results,
            "validation_passed": validate_experiment_results(statistical_results, "statistical")
        }
        
        # Overall test summary
        print(f"\n{'='*80}")
        print("COMPREHENSIVE TEST SUMMARY")
        print(f"{'='*80}")
        
        total_tests = len(test_results)
        passed_tests = sum(1 for test in test_results.values() if test["validation_passed"])
        
        print(f"Total tests run: {total_tests}")
        print(f"Tests passed: {passed_tests}")
        print(f"Tests failed: {total_tests - passed_tests}")
        print(f"Success rate: {passed_tests/total_tests*100:.1f}%")
        
        for test_name, test_data in test_results.items():
            status = "✅ PASSED" if test_data["validation_passed"] else "❌ FAILED"
            print(f"  {test_name.upper()}: {status}")
        
        # Check if Strands agent was used
        strands_used = any(
            result.get("results", {}).get("execution_method") == "strands_agent" 
            for result in test_results.values()
        )
        
        if strands_used:
            print("\n🎉 Strands SDK integration working correctly!")
        else:
            print("\n⚠️  Using fallback mode (direct tools) - check Strands configuration")
        
        print(f"\nTest completed at: {datetime.now().isoformat()}")
        
        return passed_tests == total_tests
        
    except Exception as e:
        logger.error(f"Comprehensive test failed: {str(e)}")
        print(f"\n❌ COMPREHENSIVE TEST FAILED: {str(e)}")
        return False


async def main():
    """Main test function with command line argument support"""
    parser = argparse.ArgumentParser(description="Test Strands Experiment Agent")
    parser.add_argument(
        "--experiment", 
        choices=["classification", "regression", "statistical", "comprehensive"],
        default="comprehensive",
        help="Type of experiment to run"
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="Custom session ID for the test"
    )
    
    args = parser.parse_args()
    
    # Generate session ID if not provided
    if args.session_id is None:
        timestamp = int(datetime.now().timestamp())
        args.session_id = f"experiment_test_{timestamp}_{args.experiment}"
    
    # Initialize components
    shared_memory = SharedMemory()
    agent = create_experiment_agent(shared_memory)
    
    try:
        if args.experiment == "comprehensive":
            success = await run_comprehensive_test()
            sys.exit(0 if success else 1)
        elif args.experiment == "classification":
            results = await test_classification_experiment(agent, args.session_id)
            success = validate_experiment_results(results, "classification")
        elif args.experiment == "regression":
            results = await test_regression_experiment(agent, args.session_id)
            success = validate_experiment_results(results, "regression")
        elif args.experiment == "statistical":
            results = await test_statistical_analysis(agent, args.session_id)
            success = validate_experiment_results(results, "statistical")
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
        print(f"\n❌ TEST FAILED: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())