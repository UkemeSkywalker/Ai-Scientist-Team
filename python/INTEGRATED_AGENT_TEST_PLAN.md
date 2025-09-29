# Integrated Agent Test Plan

## Overview
Create a comprehensive test that demonstrates proper multi-agent coordination between Data Agent and Experiment Agent using real datasets (no mock data).

## Current State Analysis

### ✅ Working Components
- **Data Agent**: Fully functional with smart dataset discovery
- **Experiment Agent**: Functional but missing Data Agent integration
- **test_strands_smart_data.py**: Tests Data Agent in isolation successfully

### ❌ Missing Integration
- **test_strands_experiment.py**: Uses hardcoded fake data contexts
- **No agent coordination**: Experiment Agent doesn't call Data Agent
- **Broken data flow**: Experiment tools expect datasets that don't exist

## Execution Process

### Phase 1: Data Collection (Use Existing Test)
```bash
# Test Data Agent first to verify it works
python test_strands_smart_data.py --search "iris classification machine learning"
```

**Expected Output:**
- Data Agent finds real iris datasets from Kaggle/HuggingFace
- Processes and cleans datasets
- Stores in S3 with category-based organization
- Returns S3 locations and metadata

### Phase 2: Create Integrated Test Script

**File**: `test_integrated_agents.py`

**Flow:**
1. **Data Collection Phase**
   ```python
   # Step 1: Initialize both agents
   data_agent = DataAgent(shared_memory)
   experiment_agent = ExperimentAgent(shared_memory)
   
   # Step 2: Get real datasets
   data_result = await data_agent.execute_data_collection(
       query="iris classification machine learning",
       session_id=session_id
   )
   ```

2. **Data Context Extraction**
   ```python
   # Step 3: Extract real S3 locations and metadata
   s3_locations = data_result.get("s3_locations", [])
   processed_datasets = data_result.get("processed_datasets", [])
   
   # Step 4: Create real data context from actual results
   real_data_context = {
       "datasets": [
           {
               "name": dataset["original_dataset"]["name"],
               "s3_location": dataset["storage_results"]["s3_location"],
               "columns": dataset["original_dataset"].get("columns", []),
               "target": dataset["original_dataset"].get("target"),
               "shape": dataset["original_dataset"].get("shape", []),
               "data_quality": dataset["cleaning_results"]["quality_metrics"]["overall_score"]
           }
           for dataset in processed_datasets
       ],
       "category": data_result.get("category"),
       "preprocessing": {"scaling": "standard", "train_test_split": 0.2}
   }
   ```

3. **Experiment Execution**
   ```python
   # Step 5: Pass real data to Experiment Agent
   experiment_result = await experiment_agent.execute_experiments(
       hypotheses=create_classification_hypotheses(),
       data_context=json.dumps(real_data_context),
       session_id=session_id
   )
   ```

### Phase 3: Validation

**Success Criteria:**
- Data Agent finds and processes real datasets
- S3 locations contain actual data files
- Experiment Agent successfully loads data from S3
- SageMaker training jobs execute with real data
- Statistical analysis runs on actual datasets
- End-to-end pipeline works without mock data

## Key Integration Points

### 1. Shared Memory Coordination
```python
shared_memory = SharedMemory()
# Both agents share context through session_id
```

### 2. Real Data Flow
```python
# Data Agent output becomes Experiment Agent input
data_context = extract_real_data_context(data_agent_results)
experiment_results = await experiment_agent.execute_experiments(hypotheses, data_context, session_id)
```

### 3. S3 Integration
```python
# Real S3 paths from Data Agent
s3_location = {
    "bucket": "ai-scientist-team-data",
    "key": "datasets/machine-learning/iris_dataset/20241228_143022/processed_data.json",
    "category_path": "datasets/machine-learning/"
}
```

## Expected Benefits

### 1. **Real Multi-Agent Coordination**
- Demonstrates proper agent handoff
- Shows shared memory usage
- Validates end-to-end workflow

### 2. **No Mock Data Dependencies**
- Uses actual Kaggle/HuggingFace datasets
- Real S3 storage and retrieval
- Authentic ML training pipeline

### 3. **Production-Ready Testing**
- Tests actual AWS integrations
- Validates SageMaker functionality
- Demonstrates scalable architecture

## Implementation Steps

1. **Run existing Data Agent test** to verify functionality
2. **Create `test_integrated_agents.py`** with proper agent coordination
3. **Test data flow** from Data Agent → Experiment Agent
4. **Validate results** ensure no mock data is used
5. **Document findings** and create usage examples

## Success Metrics

- ✅ Data Agent finds real datasets
- ✅ Datasets stored in S3 successfully
- ✅ Experiment Agent loads real data
- ✅ SageMaker training completes
- ✅ Statistical analysis produces results
- ✅ End-to-end pipeline works
- ✅ No mock data used anywhere

This plan ensures proper multi-agent coordination with real data flow, demonstrating the full AI Scientist Team capability.