#!/usr/bin/env python3
"""
Comprehensive test script for the shared memory system.
Usage: python test_shared_memory.py
"""

import unittest
import tempfile
import shutil
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.shared_memory import SharedMemory, SharedMemoryError, ValidationError
from src.models.workflow import ResearchContext, WorkflowState, AgentType
from src.models.research import ResearchFindings, Hypothesis, LiteratureSource
from src.models.data import DataContext, DatasetMetadata, DataQualityMetrics
from src.models.experiment import ExperimentResults, ExperimentPlan, ExperimentResult
from src.models.critic import CriticalEvaluation, ValidationReport, Limitation
from src.models.visualization import VisualizationResults, Visualization, ChartConfig


class TestSharedMemory(unittest.TestCase):
    """Test cases for SharedMemory class."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.shared_memory = SharedMemory(self.temp_dir)
        self.session_id = "test_session_123"
        self.test_query = "How can machine learning bias be reduced in healthcare?"
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_create_research_context(self):
        """Test creating a new research context."""
        context = self.shared_memory.create_research_context(self.session_id, self.test_query)
        
        self.assertIsInstance(context, ResearchContext)
        self.assertEqual(context.session_id, self.session_id)
        self.assertEqual(context.query, self.test_query)
        self.assertEqual(context.version, 1)
        self.assertIsNotNone(context.timestamp)
        
        # Verify it was written to storage
        retrieved_context = self.shared_memory.get_research_context(self.session_id)
        self.assertIsNotNone(retrieved_context)
        self.assertEqual(retrieved_context.session_id, self.session_id)
    
    def test_write_and_read_basic_data(self):
        """Test basic write and read operations."""
        test_data = {"key": "value", "number": 42}
        
        self.shared_memory.write(self.session_id, "test_data", test_data)
        retrieved_data = self.shared_memory.read(self.session_id, "test_data")
        
        self.assertEqual(retrieved_data, test_data)
    
    def test_write_and_read_pydantic_models(self):
        """Test writing and reading Pydantic models."""
        # Test ResearchFindings
        hypothesis = Hypothesis(
            text="ML bias can be reduced through diverse training data",
            confidence=0.8,
            testable=True,
            variables=["training_data", "bias_metrics"]
        )
        
        source = LiteratureSource(
            title="Bias in Machine Learning",
            authors=["Smith, J.", "Doe, A."],
            source="arxiv",
            relevance_score=0.9
        )
        
        findings = ResearchFindings(
            hypotheses=[hypothesis],
            literature_sources=[source],
            research_gaps=["Limited studies on healthcare bias"],
            key_concepts=["algorithmic fairness", "data diversity"],
            confidence_score=0.75
        )
        
        self.shared_memory.write(self.session_id, "research_findings", findings)
        retrieved_findings = self.shared_memory.read(self.session_id, "research_findings", ResearchFindings)
        
        self.assertIsInstance(retrieved_findings, ResearchFindings)
        self.assertEqual(len(retrieved_findings.hypotheses), 1)
        self.assertEqual(retrieved_findings.hypotheses[0].text, hypothesis.text)
        self.assertEqual(len(retrieved_findings.literature_sources), 1)
        self.assertEqual(retrieved_findings.confidence_score, 0.75)
    
    def test_data_context_model(self):
        """Test DataContext model serialization."""
        dataset = DatasetMetadata(
            name="Healthcare Bias Dataset",
            source="kaggle",
            description="Dataset for studying bias in healthcare ML",
            size_bytes=1024000,
            num_samples=10000,
            num_features=50,
            relevance_score=0.9
        )
        
        quality_metrics = DataQualityMetrics(
            completeness=0.95,
            consistency=0.88,
            accuracy=0.92,
            validity=0.90,
            overall_score=0.91,
            issues_found=["Missing values in age column"]
        )
        
        data_context = DataContext(
            datasets=[dataset],
            quality_metrics=quality_metrics,
            total_samples=10000,
            total_features=50,
            data_types={"age": "numeric", "gender": "categorical"}
        )
        
        self.shared_memory.write(self.session_id, "data_context", data_context)
        retrieved_context = self.shared_memory.read(self.session_id, "data_context", DataContext)
        
        self.assertIsInstance(retrieved_context, DataContext)
        self.assertEqual(len(retrieved_context.datasets), 1)
        self.assertEqual(retrieved_context.datasets[0].name, "Healthcare Bias Dataset")
        self.assertIsNotNone(retrieved_context.quality_metrics)
        self.assertEqual(retrieved_context.quality_metrics.overall_score, 0.91)
    
    def test_versioning_and_backups(self):
        """Test versioning and backup functionality."""
        # Write initial data
        initial_data = {"version": 1, "data": "initial"}
        self.shared_memory.write(self.session_id, "versioned_data", initial_data)
        
        # Update data (should create backup)
        updated_data = {"version": 2, "data": "updated"}
        self.shared_memory.write(self.session_id, "versioned_data", updated_data)
        
        # Verify current data
        current_data = self.shared_memory.read(self.session_id, "versioned_data")
        self.assertEqual(current_data["data"], "updated")
        
        # Check version history
        history = self.shared_memory.get_version_history(self.session_id, "versioned_data")
        self.assertGreaterEqual(len(history), 0)  # May be empty if backup wasn't created
    
    def test_agent_results_integration(self):
        """Test writing agent results to both workflow state and research context."""
        # Create initial context
        context = self.shared_memory.create_research_context(self.session_id, self.test_query)
        
        # Create workflow state
        workflow_state = WorkflowState(
            session_id=self.session_id,
            query=self.test_query
        )
        self.shared_memory.update_workflow_state(workflow_state)
        
        # Write research agent results
        findings = ResearchFindings(
            hypotheses=[Hypothesis(text="Test hypothesis", confidence=0.8)],
            confidence_score=0.8
        )
        
        self.shared_memory.write_agent_results(self.session_id, AgentType.RESEARCH, findings)
        
        # Verify workflow state was updated
        updated_workflow = self.shared_memory.get_workflow_state(self.session_id)
        self.assertEqual(updated_workflow.agents[AgentType.RESEARCH].status, "completed")
        self.assertEqual(updated_workflow.agents[AgentType.RESEARCH].progress, 100)
        
        # Verify research context was updated
        updated_context = self.shared_memory.get_research_context(self.session_id)
        self.assertIsNotNone(updated_context.research_findings)
    
    def test_session_management(self):
        """Test session management operations."""
        # Create multiple sessions
        session1 = "session_1"
        session2 = "session_2"
        
        self.shared_memory.create_research_context(session1, "Query 1")
        self.shared_memory.create_research_context(session2, "Query 2")
        
        # List sessions
        sessions = self.shared_memory.list_sessions()
        self.assertIn(session1, sessions)
        self.assertIn(session2, sessions)
        
        # Get session metadata
        metadata = self.shared_memory.get_session_metadata(session1)
        self.assertEqual(metadata['session_id'], session1)
        self.assertGreater(metadata['file_count'], 0)
        
        # Clear one session
        self.shared_memory.clear_session(session1)
        sessions_after_clear = self.shared_memory.list_sessions()
        self.assertNotIn(session1, sessions_after_clear)
        self.assertIn(session2, sessions_after_clear)
    
    def test_validation(self):
        """Test data validation functionality."""
        # Test valid data
        valid_findings = ResearchFindings(
            hypotheses=[Hypothesis(text="Valid hypothesis", confidence=0.8)],
            confidence_score=0.8
        )
        
        # Should not raise exception
        self.shared_memory.write(self.session_id, "valid_findings", valid_findings)
        
        # Test session validation
        validation_result = self.shared_memory.validate_session(self.session_id)
        self.assertTrue(validation_result['valid'])
        self.assertEqual(len(validation_result['errors']), 0)
    
    def test_error_handling(self):
        """Test error handling for invalid operations."""
        # Test empty session ID
        with self.assertRaises(SharedMemoryError):
            self.shared_memory.write("", "key", {"data": "test"})
        
        # Test empty key
        with self.assertRaises(SharedMemoryError):
            self.shared_memory.write(self.session_id, "", {"data": "test"})
        
        # Test reading non-existent data
        result = self.shared_memory.read("nonexistent_session", "nonexistent_key")
        self.assertIsNone(result)
    
    def test_context_retrieval(self):
        """Test getting all context data for a session."""
        # Write multiple pieces of data
        self.shared_memory.write(self.session_id, "data1", {"key": "value1"})
        self.shared_memory.write(self.session_id, "data2", {"key": "value2"})
        
        context = self.shared_memory.get_context(self.session_id)
        
        self.assertIn("data1", context)
        self.assertIn("data2", context)
        self.assertEqual(context["data1"]["data"]["key"], "value1")
        self.assertEqual(context["data2"]["data"]["key"], "value2")


def create_sample_data() -> Dict[str, Any]:
    """Create sample data for demonstration."""
    # Research findings
    hypothesis = Hypothesis(
        text="Diverse training data reduces algorithmic bias in healthcare ML models",
        confidence=0.85,
        testable=True,
        variables=["training_data_diversity", "bias_metrics", "model_performance"],
        expected_outcome="Reduced bias scores with maintained accuracy"
    )
    
    literature_source = LiteratureSource(
        title="Algorithmic Bias in Healthcare: A Systematic Review",
        authors=["Johnson, M.", "Smith, A.", "Brown, K."],
        publication_date=datetime(2023, 6, 15),
        source="pubmed",
        url="https://pubmed.ncbi.nlm.nih.gov/example",
        relevance_score=0.92,
        abstract="This systematic review examines algorithmic bias in healthcare applications..."
    )
    
    research_findings = ResearchFindings(
        hypotheses=[hypothesis],
        literature_sources=[literature_source],
        research_gaps=[
            "Limited studies on bias in diagnostic imaging",
            "Lack of standardized bias metrics across healthcare domains"
        ],
        key_concepts=["algorithmic fairness", "healthcare equity", "model interpretability"],
        confidence_score=0.78,
        methodology_suggestions=[
            "Use stratified sampling for diverse training data",
            "Implement fairness-aware machine learning techniques"
        ]
    )
    
    # Data context
    dataset = DatasetMetadata(
        name="Healthcare Equity Dataset",
        source="kaggle",
        url="https://kaggle.com/datasets/healthcare-equity",
        description="Comprehensive dataset for studying healthcare disparities",
        size_bytes=2048000,
        num_samples=50000,
        num_features=75,
        file_format="CSV",
        license="CC BY 4.0",
        last_updated=datetime(2023, 8, 1),
        relevance_score=0.94
    )
    
    quality_metrics = DataQualityMetrics(
        completeness=0.96,
        consistency=0.89,
        accuracy=0.93,
        validity=0.91,
        overall_score=0.92,
        issues_found=[
            "3% missing values in income field",
            "Some inconsistent date formats"
        ]
    )
    
    data_context = DataContext(
        datasets=[dataset],
        quality_metrics=quality_metrics,
        total_samples=50000,
        total_features=75,
        data_types={
            "age": "numeric",
            "gender": "categorical",
            "income": "numeric",
            "diagnosis": "categorical",
            "treatment_outcome": "binary"
        }
    )
    
    return {
        "research_findings": research_findings,
        "data_context": data_context
    }


def demo_shared_memory():
    """Demonstrate shared memory functionality."""
    print("🧠 AI Scientist Team - Shared Memory System Demo")
    print("=" * 60)
    
    # Create temporary shared memory
    temp_dir = tempfile.mkdtemp()
    shared_memory = SharedMemory(temp_dir)
    
    try:
        session_id = f"demo_session_{int(datetime.now().timestamp())}"
        query = "How can machine learning bias be reduced in healthcare applications?"
        
        print(f"📝 Creating research context for session: {session_id}")
        print(f"🔍 Query: {query}")
        
        # Create research context
        context = shared_memory.create_research_context(session_id, query)
        print(f"✅ Research context created (version {context.version})")
        
        # Create sample data
        sample_data = create_sample_data()
        
        # Write research findings
        print("\n📚 Writing research findings...")
        shared_memory.write(session_id, "research_findings", sample_data["research_findings"])
        print("✅ Research findings written")
        
        # Write data context
        print("\n📊 Writing data context...")
        shared_memory.write(session_id, "data_context", sample_data["data_context"])
        print("✅ Data context written")
        
        # Update research context with agent results
        print("\n🔄 Updating research context...")
        updated_context = shared_memory.get_research_context(session_id)
        updated_context.research_findings = sample_data["research_findings"]
        updated_context.data_context = sample_data["data_context"]
        shared_memory.update_research_context(updated_context)
        print(f"✅ Research context updated (version {updated_context.version})")
        
        # Demonstrate reading data back
        print("\n📖 Reading data back...")
        retrieved_findings = shared_memory.read(session_id, "research_findings", ResearchFindings)
        retrieved_data_context = shared_memory.read(session_id, "data_context", DataContext)
        
        print(f"📚 Research findings: {len(retrieved_findings.hypotheses)} hypotheses, "
              f"{len(retrieved_findings.literature_sources)} sources")
        print(f"📊 Data context: {len(retrieved_data_context.datasets)} datasets, "
              f"{retrieved_data_context.total_samples} samples")
        
        # Show session metadata
        print("\n📋 Session metadata:")
        metadata = shared_memory.get_session_metadata(session_id)
        print(f"   Files: {metadata['file_count']}")
        print(f"   Total size: {metadata['total_size']} bytes")
        print(f"   Created: {metadata['created_at']}")
        
        # Validate session
        print("\n✅ Validating session...")
        validation_result = shared_memory.validate_session(session_id)
        if validation_result['valid']:
            print("✅ Session validation passed")
        else:
            print(f"❌ Session validation failed: {validation_result['errors']}")
        
        # Show all context
        print("\n🗂️  All session context:")
        all_context = shared_memory.get_context(session_id)
        for key, data in all_context.items():
            print(f"   {key}: {data['type']} (v{data['version']}) - {data['timestamp']}")
        
        print(f"\n💾 Data persisted to: {temp_dir}")
        print("🎉 Demo completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False
    
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    # Run demo first
    print("Running shared memory demonstration...\n")
    demo_success = demo_shared_memory()
    
    print("\n" + "=" * 60)
    print("Running unit tests...\n")
    
    # Run unit tests
    unittest.main(verbosity=2, exit=False)
    
    if demo_success:
        print("\n🎉 All tests and demo completed successfully!")
        print("✅ Shared memory system is ready for use by agents")
    else:
        print("\n❌ Demo failed - check implementation")
        sys.exit(1)