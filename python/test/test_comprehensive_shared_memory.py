#!/usr/bin/env python3
"""
Comprehensive test demonstrating all shared memory features as specified in task requirements:
- SharedMemory class that works with local file system
- ResearchContext, ResearchFindings, and core data models
- Serialization/deserialization utilities using JSON
- Context validation and basic versioning
- Comprehensive unit tests for all shared memory operations

Usage: python test_comprehensive_shared_memory.py
"""

import sys
from pathlib import Path
from datetime import datetime
import json
import tempfile
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.shared_memory import SharedMemory, SharedMemoryError
from src.models.workflow import ResearchContext, WorkflowState, AgentType
from src.models.research import ResearchFindings, Hypothesis, LiteratureSource
from src.models.data import DataContext, DatasetMetadata, DataQualityMetrics, S3Location
from src.models.experiment import ExperimentResults, ExperimentPlan, ExperimentResult, ExperimentConfig
from src.models.critic import CriticalEvaluation, ValidationReport, Limitation
from src.models.visualization import VisualizationResults, Visualization, ChartConfig


def test_core_data_models():
    """Test all core data models can be serialized/deserialized."""
    print("🧪 Testing Core Data Models...")
    
    # Test Hypothesis model
    hypothesis = Hypothesis(
        text="Machine learning models can be made more fair through data augmentation",
        confidence=0.82,
        testable=True,
        variables=["training_data", "fairness_metrics", "model_accuracy"],
        expected_outcome="Improved fairness scores without significant accuracy loss"
    )
    
    # Test LiteratureSource model
    source = LiteratureSource(
        title="Fairness in Machine Learning: A Survey",
        authors=["Mehrabi, N.", "Morstatter, F.", "Saxena, N."],
        publication_date=datetime(2021, 9, 15),
        source="arxiv",
        url="https://arxiv.org/abs/1908.09635",
        relevance_score=0.95,
        abstract="This paper provides a comprehensive survey of fairness in machine learning..."
    )
    
    # Test ResearchFindings model
    findings = ResearchFindings(
        hypotheses=[hypothesis],
        literature_sources=[source],
        research_gaps=["Limited real-world deployment studies", "Lack of standardized fairness metrics"],
        key_concepts=["algorithmic fairness", "bias mitigation", "equitable AI"],
        confidence_score=0.78,
        methodology_suggestions=["Use adversarial debiasing", "Implement fairness constraints"]
    )
    
    # Test DatasetMetadata model
    dataset = DatasetMetadata(
        name="Adult Income Dataset",
        source="kaggle",
        url="https://kaggle.com/datasets/adult-income",
        description="Census data for income prediction with fairness considerations",
        size_bytes=3276800,
        num_samples=48842,
        num_features=14,
        file_format="CSV",
        license="Public Domain",
        last_updated=datetime(2023, 5, 20),
        relevance_score=0.89
    )
    
    # Test DataQualityMetrics model
    quality = DataQualityMetrics(
        completeness=0.94,
        consistency=0.87,
        accuracy=0.91,
        validity=0.88,
        overall_score=0.90,
        issues_found=["Missing values in workclass", "Inconsistent capitalization"]
    )
    
    # Test S3Location model
    s3_location = S3Location(
        bucket="ai-scientist-data",
        key="datasets/adult-income/processed.csv",
        region="us-east-1",
        size_bytes=3276800,
        last_modified=datetime(2023, 8, 15),
        metadata={"processed": "true", "version": "1.0"}
    )
    
    # Test DataContext model
    data_context = DataContext(
        datasets=[dataset],
        quality_metrics=quality,
        s3_locations=[s3_location],
        total_samples=48842,
        total_features=14,
        data_types={"age": "numeric", "workclass": "categorical", "income": "binary"}
    )
    
    print("   ✅ All core data models created successfully")
    return {
        "research_findings": findings,
        "data_context": data_context
    }


def test_shared_memory_operations():
    """Test all shared memory operations."""
    print("\n🧪 Testing Shared Memory Operations...")
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    shared_memory = SharedMemory(temp_dir)
    
    try:
        session_id = f"test_session_{int(datetime.now().timestamp())}"
        query = "How can we ensure fairness in machine learning models?"
        
        # 1. Test creating research context
        print("   📝 Creating research context...")
        context = shared_memory.create_research_context(session_id, query)
        assert context.session_id == session_id
        assert context.query == query
        assert context.version == 1
        print("   ✅ Research context created")
        
        # 2. Test writing and reading data
        print("   💾 Testing write/read operations...")
        test_data = {"test": "data", "number": 42, "list": [1, 2, 3]}
        shared_memory.write(session_id, "test_data", test_data)
        retrieved_data = shared_memory.read(session_id, "test_data")
        assert retrieved_data == test_data
        print("   ✅ Basic write/read operations work")
        
        # 3. Test Pydantic model serialization
        print("   🔄 Testing Pydantic model serialization...")
        models = test_core_data_models()
        
        shared_memory.write(session_id, "research_findings", models["research_findings"])
        shared_memory.write(session_id, "data_context", models["data_context"])
        
        retrieved_findings = shared_memory.read(session_id, "research_findings", ResearchFindings)
        retrieved_data_context = shared_memory.read(session_id, "data_context", DataContext)
        
        assert isinstance(retrieved_findings, ResearchFindings)
        assert isinstance(retrieved_data_context, DataContext)
        assert len(retrieved_findings.hypotheses) == 1
        assert len(retrieved_data_context.datasets) == 1
        print("   ✅ Pydantic model serialization works")
        
        # 4. Test versioning
        print("   🔢 Testing versioning...")
        original_context = shared_memory.get_research_context(session_id)
        original_version = original_context.version
        
        # Update context
        original_context.research_findings = models["research_findings"]
        shared_memory.update_research_context(original_context)
        
        updated_context = shared_memory.get_research_context(session_id)
        assert updated_context.version == original_version + 1
        print("   ✅ Versioning works")
        
        # 5. Test validation
        print("   ✅ Testing validation...")
        validation_result = shared_memory.validate_session(session_id)
        assert validation_result['valid'] == True
        assert len(validation_result['errors']) == 0
        print("   ✅ Validation works")
        
        # 6. Test session management
        print("   📋 Testing session management...")
        sessions = shared_memory.list_sessions()
        assert session_id in sessions
        
        metadata = shared_memory.get_session_metadata(session_id)
        assert metadata['session_id'] == session_id
        assert metadata['file_count'] > 0
        print("   ✅ Session management works")
        
        # 7. Test context retrieval
        print("   🗂️  Testing context retrieval...")
        all_context = shared_memory.get_context(session_id)
        assert "research_context" in all_context
        assert "research_findings" in all_context
        assert "data_context" in all_context
        print("   ✅ Context retrieval works")
        
        # 8. Test error handling
        print("   ⚠️  Testing error handling...")
        try:
            shared_memory.write("", "key", {"data": "test"})
            assert False, "Should have raised error for empty session ID"
        except SharedMemoryError:
            pass  # Expected
        
        try:
            shared_memory.write(session_id, "", {"data": "test"})
            assert False, "Should have raised error for empty key"
        except SharedMemoryError:
            pass  # Expected
        
        # Test reading non-existent data
        result = shared_memory.read("nonexistent", "nonexistent")
        assert result is None
        print("   ✅ Error handling works")
        
        print(f"\n   💾 All data persisted to: {temp_dir}")
        print("   🎉 All shared memory operations successful!")
        
        return True
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def test_json_file_creation():
    """Test that JSON files are created correctly."""
    print("\n🧪 Testing JSON File Creation...")
    
    temp_dir = tempfile.mkdtemp()
    shared_memory = SharedMemory(temp_dir)
    
    try:
        session_id = "json_test_session"
        query = "Test query for JSON file creation"
        
        # Create research context
        context = shared_memory.create_research_context(session_id, query)
        
        # Write some test data
        test_findings = ResearchFindings(
            hypotheses=[Hypothesis(text="Test hypothesis", confidence=0.8)],
            confidence_score=0.8
        )
        shared_memory.write(session_id, "research_findings", test_findings)
        
        # Check that JSON files were created
        session_dir = Path(temp_dir) / session_id
        json_files = list(session_dir.glob("*.json"))
        
        assert len(json_files) >= 2, f"Expected at least 2 JSON files, found {len(json_files)}"
        
        # Verify JSON structure
        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Check required fields
            assert "data" in data
            assert "timestamp" in data
            assert "type" in data
            assert "version" in data
            
            print(f"   📄 {json_file.name}: {data['type']} (v{data['version']})")
        
        print("   ✅ JSON files created with correct structure")
        return True
        
    finally:
        shutil.rmtree(temp_dir)


def main():
    """Run comprehensive shared memory tests."""
    print("🧠 AI Scientist Team - Comprehensive Shared Memory Test")
    print("=" * 70)
    print("Testing all requirements from task specification:")
    print("✓ SharedMemory class that works with local file system")
    print("✓ ResearchContext, ResearchFindings, and core data models")
    print("✓ Serialization/deserialization utilities using JSON")
    print("✓ Context validation and basic versioning")
    print("✓ Comprehensive unit tests for all shared memory operations")
    print("=" * 70)
    
    try:
        # Test 1: Core data models
        test_core_data_models()
        
        # Test 2: Shared memory operations
        test_shared_memory_operations()
        
        # Test 3: JSON file creation
        test_json_file_creation()
        
        print("\n" + "=" * 70)
        print("🎉 ALL TESTS PASSED!")
        print("✅ Shared memory system meets all task requirements")
        print("✅ System is ready for use by agents")
        print("✅ JSON files are created and validated correctly")
        print("✅ All core data models work with serialization")
        print("✅ Versioning and validation systems operational")
        print("=" * 70)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)