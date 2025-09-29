# Shared Memory Integration Plan
## Research Agent → Data Agent → Experiment Agent Workflow

### Overview
This plan eliminates hardcoded hypotheses and datasets by implementing a complete shared memory-based workflow where each agent reads from and writes to shared memory, enabling seamless data flow between agents.

## Current Issues to Fix

### 1. Hypothesis-Data Mismatch
- **Problem**: Test script uses hardcoded iris hypotheses with sentiment datasets
- **Solution**: Research Agent generates query-appropriate hypotheses

### 2. S3 Path Structure Issues  
- **Problem**: Experiment design tool generates wrong S3 paths (`s3://...datasets/tabular/iris/...`)
- **Actual**: Real paths are `s3://ai-scientist-team-data-unique-2024/datasets/natural-language-processing/...`
- **Solution**: Use actual S3 paths from shared memory

### 3. Pydantic Model Validation
- **Problem**: Training tool expects dict `.get()` method but receives Pydantic models
- **Solution**: Convert Pydantic models to dict in experiment design tool

## Implementation Plan

### Phase 1: Research Agent Integration
**Objective**: Generate query-appropriate hypotheses and store in shared memory

**Changes Required:**
1. **Update test script** to start with Research Agent
2. **Generate hypotheses** using `research_agent.execute_research(query, session_id)`
3. **Store results** in shared memory with key `"research_result"`

**Code Changes:**
```python
# In test_integrated_agents.py
research_agent = create_research_agent(shared_memory)
research_result = await research_agent.execute_research(query, session_id)
```

### Phase 2: Data Agent Integration
**Objective**: Data Agent reads research context from shared memory

**Changes Required:**
1. **Read research context** from shared memory in Data Agent
2. **Extract query** from research results instead of parameter
3. **Maintain existing** S3 storage functionality

**Code Changes:**
```python
# In data_agent.py execute_data_collection()
research_results = self.shared_memory.read(session_id, "research_result")
if research_results:
    query_from_research = research_results.get("query", query)
    # Can also access hypotheses if needed:
    # hypotheses = research_results.get("hypotheses", [])
```

### Phase 3: Experiment Agent Integration  
**Objective**: Read both research and data results from shared memory

**Changes Required:**
1. **Read research results** for hypotheses
2. **Read data results** for context (already implemented)
3. **Remove dependency** on passed parameters

**Code Changes:**
```python
# In experiment_agent.py execute_experiments()
# Read research results for hypotheses
research_results = self.shared_memory.read(session_id, "research_result")
if research_results:
    hypotheses_to_use = json.dumps(research_results.get("hypotheses", []))

# Read data results for S3 paths (already implemented)
data_context = self._extract_data_context_from_shared_memory(session_id)
```

### Phase 4: Fix S3 Path Structure
**Objective**: Use actual S3 paths from shared memory instead of hardcoded paths

**Changes Required:**
1. **Update experiment design tool** to use real S3 paths from data context
2. **Remove hardcoded** iris path generation
3. **Ensure S3 paths** are correctly extracted from shared memory

**Code Changes:**
```python
# In experiment_tools.py experiment_design_tool()
# Extract S3 location from data context (nested in datasets array)
if data_context and 'datasets' in data_context:
    for dataset in data_context['datasets']:
        s3_location = dataset.get('s3_location')
        if s3_location and s3_location.startswith('s3://'):
            experiment_config["direct_s3_paths"] = [s3_location]
            break
```

### Phase 5: Fix Pydantic Model Issues
**Objective**: Ensure training tool receives plain dictionaries

**Changes Required:**
1. **Convert Pydantic models to dict** in experiment design tool
2. **Return JSON dict** instead of Pydantic model
3. **Ensure compatibility** with training tool's `.get()` method calls

**Code Changes:**
```python
# In experiment_tools.py experiment_design_tool()
plan_dict = plan.model_dump()  # Convert Pydantic to dict
return json.dumps(plan_dict)   # Return as JSON dict
```

## Shared Memory Data Flow

### Storage Structure
```
data/shared_memory/{session_id}/
├── research_result.json     # Research Agent output with hypotheses
├── data_result.json        # Data Agent output with S3 paths
├── experiment_result.json  # Experiment Agent output
└── backups/               # Version history
    ├── research_result_v1.json
    └── data_result_v1.json
```

### File Format (JSON with metadata wrapper)
```json
{
  "data": {                    # Actual agent results
    "query": "research query",
    "hypotheses": [...],       # For research_result
    "datasets": [...],         # For data_result
    "s3_location": "s3://...", # For data_result
    "status": "success"
  },
  "timestamp": "2024-01-01T12:00:00",
  "type": "dict",
  "version": 1,
  "key": "research_result",
  "session_id": "session_123"
}
```

### Data Flow Sequence
1. **Research Agent** → stores complete results in `research_result.json`
2. **Data Agent** reads `research_result.json` → writes `data_result.json`
3. **Experiment Agent** reads both files → writes `experiment_result.json`

### Key-Value Mapping & Access Patterns
- `"research_result"` → Research findings with nested hypotheses
  - Access: `research_data = shared_memory.read(session_id, "research_result")`
  - Hypotheses: `research_data.get("hypotheses", [])`
- `"data_result"` → Dataset information with S3 paths
  - Access: `data_context = shared_memory.read(session_id, "data_result")`
  - S3 paths: `data_context.get("datasets", [])[0].get("s3_location")`
- `"experiment_result"` → ML training and analysis results

## Updated Test Script Flow

### New Workflow
```python
async def test_integrated_workflow(query: str, session_id: str):
    shared_memory = SharedMemory()
    
    # Phase 1: Research Agent
    research_agent = create_research_agent(shared_memory)
    research_result = await research_agent.execute_research(query, session_id)
    
    # Phase 2: Data Agent (reads from shared memory)
    data_agent = create_data_agent(shared_memory)  
    data_result = await data_agent.execute_data_collection(query, session_id)
    
    # Phase 3: Experiment Agent (reads from shared memory)
    experiment_agent = create_experiment_agent(shared_memory)
    experiment_result = await experiment_agent.execute_experiments("", "", session_id)
```

### Removed Dependencies
- ❌ Hardcoded iris hypotheses
- ❌ Manual data context extraction
- ❌ Parameter passing between agents
- ❌ Hardcoded S3 paths

### Added Benefits
- ✅ Query-appropriate hypotheses
- ✅ Real S3 path usage
- ✅ Complete shared memory communication
- ✅ No hardcoded data dependencies

## Expected Outcomes

### Integration Test Results
- **Research Agent**: Generates sentiment-appropriate hypotheses
- **Data Agent**: Finds sentiment datasets and stores in S3
- **Experiment Agent**: Uses real hypotheses with real S3 data
- **Training Tool**: Successfully loads data from correct S3 paths
- **Overall**: End-to-end workflow with no hardcoded dependencies

### Performance Improvements
- **Hypothesis Alignment**: Hypotheses match actual datasets
- **S3 Path Accuracy**: Training uses correct S3 locations
- **Type Compatibility**: Dict interface works with training tool
- **Error Reduction**: Eliminates mismatch-related failures

## Implementation Priority
1. **High Priority**: Fix S3 path structure (immediate training fix)
2. **High Priority**: Convert Pydantic to dict (immediate compatibility fix)
3. **Medium Priority**: Research Agent integration (workflow improvement)
4. **Low Priority**: Remove parameter dependencies (cleanup)

## Shared Memory Implementation Details

### Storage Mechanism
- **File-based storage**: Each session gets its own directory under `data/shared_memory/{session_id}/`
- **JSON format**: All data stored as JSON with metadata wrapper (timestamp, version, type)
- **Atomic writes**: Uses temporary files to prevent corruption during writes
- **Versioning**: Automatic backup creation before overwrites in `backups/` folder

### Agent Communication Pattern
```python
# Research Agent stores results
self.shared_memory.update_context(session_id, {
    "research_result": {
        "query": query,
        "hypotheses": generated_hypotheses,
        "literature_summary": {...},
        "status": "success"
    }
})

# Data Agent reads research context
research_data = self.shared_memory.read(session_id, "research_result")
query_from_research = research_data.get("query") if research_data else query

# Experiment Agent reads both research and data results
research_results = self.shared_memory.read(session_id, "research_result")
data_results = self.shared_memory.read(session_id, "data_result")
hypotheses = research_results.get("hypotheses", []) if research_results else []
s3_paths = [d.get("s3_location") for d in data_results.get("datasets", [])] if data_results else []
```

### Error Handling
- **Graceful fallbacks**: Agents continue with default values if shared memory reads fail
- **Validation**: Optional Pydantic model validation on read operations
- **Backup recovery**: Can restore previous versions if current data is corrupted

This plan creates a fully integrated, shared memory-based workflow that eliminates all hardcoded dependencies and ensures proper data flow between agents.