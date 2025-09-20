"""
Mock tools for Strands agents - used for initial testing and development
"""

import json
import time
from typing import Dict, Any, List
import structlog

logger = structlog.get_logger(__name__)

def mock_research_tool(query: str) -> str:
    """
    Mock research tool that simulates literature search and hypothesis generation
    
    Args:
        query: Research query to process
        
    Returns:
        JSON string with mock research findings
    """
    logger.info("Mock research tool called", query=query)
    
    # Simulate processing time
    time.sleep(1)
    
    mock_findings = {
        "query": query,
        "hypotheses": [
            f"Hypothesis 1: {query} shows significant correlation with performance metrics",
            f"Hypothesis 2: {query} implementation varies across different domains",
            f"Hypothesis 3: {query} effectiveness depends on contextual factors"
        ],
        "literature_review": {
            "papers_found": 15,
            "key_papers": [
                {"title": f"Analysis of {query} in Modern Systems", "authors": "Smith et al.", "year": 2023},
                {"title": f"Comprehensive Study on {query}", "authors": "Johnson & Lee", "year": 2022},
                {"title": f"{query}: A Meta-Analysis", "authors": "Brown et al.", "year": 2024}
            ]
        },
        "confidence_score": 0.85,
        "status": "completed"
    }
    
    return json.dumps(mock_findings, indent=2)

def mock_data_collection_tool(dataset_type: str) -> str:
    """
    Mock data collection tool that simulates dataset discovery and processing
    
    Args:
        dataset_type: Type of dataset to search for
        
    Returns:
        JSON string with mock data collection results
    """
    logger.info("Mock data collection tool called", dataset_type=dataset_type)
    
    # Simulate processing time
    time.sleep(1.5)
    
    mock_data_results = {
        "dataset_type": dataset_type,
        "datasets_found": [
            {
                "name": f"{dataset_type}_dataset_v1",
                "source": "Kaggle",
                "size": "10,000 samples",
                "quality_score": 0.92,
                "s3_location": f"s3://ai-scientist-data/{dataset_type}/v1/"
            },
            {
                "name": f"{dataset_type}_benchmark",
                "source": "HuggingFace",
                "size": "25,000 samples",
                "quality_score": 0.88,
                "s3_location": f"s3://ai-scientist-data/{dataset_type}/benchmark/"
            }
        ],
        "preprocessing_steps": [
            "Data cleaning and validation",
            "Feature normalization",
            "Train/test split (80/20)"
        ],
        "status": "completed"
    }
    
    return json.dumps(mock_data_results, indent=2)

def mock_experiment_tool(experiment_type: str, data_context: str = "") -> str:
    """
    Mock experiment tool that simulates ML experiments and analysis
    
    Args:
        experiment_type: Type of experiment to run
        data_context: Context about available data
        
    Returns:
        JSON string with mock experiment results
    """
    logger.info("Mock experiment tool called", experiment_type=experiment_type)
    
    # Simulate processing time
    time.sleep(2)
    
    mock_experiment_results = {
        "experiment_type": experiment_type,
        "data_context": data_context,
        "experiments": [
            {
                "name": f"{experiment_type}_baseline",
                "algorithm": "Random Forest",
                "accuracy": 0.87,
                "precision": 0.85,
                "recall": 0.89,
                "f1_score": 0.87
            },
            {
                "name": f"{experiment_type}_advanced",
                "algorithm": "XGBoost",
                "accuracy": 0.92,
                "precision": 0.91,
                "recall": 0.93,
                "f1_score": 0.92
            }
        ],
        "statistical_significance": True,
        "p_value": 0.003,
        "confidence_interval": [0.89, 0.95],
        "status": "completed"
    }
    
    return json.dumps(mock_experiment_results, indent=2)

def mock_critic_tool(results_to_evaluate: str) -> str:
    """
    Mock critic tool that simulates critical evaluation of results
    
    Args:
        results_to_evaluate: Results to critically analyze
        
    Returns:
        JSON string with mock critical evaluation
    """
    logger.info("Mock critic tool called")
    
    # Simulate processing time
    time.sleep(1)
    
    mock_evaluation = {
        "evaluation_summary": "Comprehensive critical analysis completed",
        "strengths": [
            "Strong statistical significance in results",
            "Appropriate methodology for the research question",
            "Good sample size and data quality"
        ],
        "limitations": [
            "Limited generalizability to other domains",
            "Potential selection bias in data collection",
            "Need for longer-term validation studies"
        ],
        "bias_assessment": {
            "selection_bias": "Low risk",
            "confirmation_bias": "Medium risk",
            "publication_bias": "Low risk"
        },
        "confidence_score": 0.78,
        "recommendations": [
            "Conduct additional validation with different datasets",
            "Consider cross-domain evaluation",
            "Implement bias mitigation strategies"
        ],
        "status": "completed"
    }
    
    return json.dumps(mock_evaluation, indent=2)

def mock_visualization_tool(data_to_visualize: str, chart_type: str = "comprehensive") -> str:
    """
    Mock visualization tool that simulates chart and report generation
    
    Args:
        data_to_visualize: Data to create visualizations for
        chart_type: Type of visualization to create
        
    Returns:
        JSON string with mock visualization results
    """
    logger.info("Mock visualization tool called", chart_type=chart_type)
    
    # Simulate processing time
    time.sleep(1.5)
    
    mock_viz_results = {
        "chart_type": chart_type,
        "visualizations_created": [
            {
                "type": "performance_comparison",
                "format": "PNG",
                "path": "/tmp/performance_comparison.png",
                "description": "Comparison of model performance metrics"
            },
            {
                "type": "confidence_intervals",
                "format": "SVG",
                "path": "/tmp/confidence_intervals.svg",
                "description": "Statistical confidence intervals visualization"
            },
            {
                "type": "interactive_dashboard",
                "format": "HTML",
                "path": "/tmp/dashboard.html",
                "description": "Interactive results dashboard"
            }
        ],
        "report_generated": {
            "format": "PDF",
            "path": "/tmp/research_report.pdf",
            "pages": 12,
            "sections": ["Executive Summary", "Methodology", "Results", "Discussion", "Conclusions"]
        },
        "status": "completed"
    }
    
    return json.dumps(mock_viz_results, indent=2)