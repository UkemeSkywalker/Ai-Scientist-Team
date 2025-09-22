# Smart Dataset Discovery System

## Overview

The Smart Dataset Discovery System is an enhanced data management solution that automatically organizes datasets by research categories and prioritizes reusability. Instead of randomly storing datasets, it creates an intelligent, category-based structure that makes datasets easily discoverable and reusable across research projects.

## Key Features

### 🎯 Intelligent Query Categorization
- Automatically categorizes research queries into predefined domains
- Supports 10+ research categories: machine-learning, nlp, computer-vision, healthcare, finance, etc.
- Confidence scoring for categorization accuracy

### ♻️ Reusability-First Approach
- Checks existing datasets before downloading new ones
- Prioritizes high-quality existing data
- Reduces redundant downloads and storage costs
- Builds a cumulative knowledge base

### 📁 Category-Based Organization
- Organizes S3 storage by research categories
- Consistent folder structure: `datasets/{category}/{dataset_name}/{timestamp}/`
- Rich metadata for easy discovery and filtering

### 🔍 Smart Recommendations
- Combines existing and new dataset suggestions
- Relevance-based prioritization
- Quality-aware selection
- Actionable recommendations (reuse vs. download)

## Architecture

### Research Categories

```python
RESEARCH_CATEGORIES = {
    "machine-learning": ["machine learning", "ml", "neural network", "classification", ...],
    "natural-language-processing": ["nlp", "text analysis", "sentiment", ...],
    "computer-vision": ["image", "object detection", "face recognition", ...],
    "healthcare": ["medical", "clinical", "patient", "diagnosis", ...],
    "finance": ["stock", "trading", "investment", "banking", ...],
    # ... and more
}
```

### S3 Storage Structure

```
s3://ai-scientist-team-data/
├── datasets/
│   ├── machine-learning/
│   │   ├── sentiment-analysis-dataset/
│   │   │   └── 20241222_143022/
│   │   │       └── processed_data.json
│   │   └── classification-benchmark/
│   ├── natural-language-processing/
│   │   ├── text-classification-corpus/
│   │   └── language-model-data/
│   ├── computer-vision/
│   │   ├── image-classification-set/
│   │   └── object-detection-data/
│   └── healthcare/
│       ├── patient-diagnosis-data/
│       └── medical-imaging-set/
```

## Core Tools

### 1. Query Categorization
```python
from tools.data_tools import categorize_query

category, confidence = categorize_query("sentiment analysis of movie reviews")
# Returns: ("natural-language-processing", 0.85)
```

### 2. Existing Dataset Check
```python
from tools.data_tools import check_existing_datasets_tool

result = check_existing_datasets_tool("machine learning classification")
# Returns JSON with existing datasets in relevant categories
```

### 3. Smart Dataset Discovery
```python
from tools.data_tools import smart_dataset_discovery_tool

result = smart_dataset_discovery_tool("computer vision object detection", max_new_datasets=5)
# Returns comprehensive recommendations combining existing and new datasets
```

### 4. Category-Based Storage
```python
from tools.data_tools import s3_storage_tool

result = s3_storage_tool(
    dataset_info=json.dumps(dataset),
    cleaned_data_summary=json.dumps(cleaning_results),
    query="sentiment analysis"  # Used for categorization
)
# Stores in appropriate category folder with rich metadata
```

### 5. Dataset Organization
```python
from tools.data_tools import organize_dataset_categories_tool

result = organize_dataset_categories_tool(dry_run=True)
# Analyzes and reorganizes existing datasets into category structure
```

## Enhanced Data Agent

The `DataAgent` now uses smart discovery by default:

```python
from agents.data_agent import DataAgent

data_agent = DataAgent()
result = await data_agent.execute_data_collection(
    query="natural language processing for chatbots",
    session_id="session_001"
)

# Result includes:
# - Category classification
# - Existing datasets reused
# - New datasets added
# - Organization benefits
# - Reusability metrics
```

## Workflow

### 1. Smart Discovery Process
1. **Query Analysis**: Categorize the research query
2. **Existing Check**: Search for relevant datasets in the category
3. **Gap Analysis**: Determine if new datasets are needed
4. **Source Search**: Search Kaggle, HuggingFace if gaps exist
5. **Recommendation**: Prioritize existing high-quality datasets
6. **Processing**: Clean and store new datasets in category structure

### 2. Recommendation Types
- **Existing High Priority**: High-quality datasets in same category
- **Existing Medium Priority**: Related datasets from other categories
- **New High Priority**: Highly relevant new datasets from Kaggle
- **New Medium Priority**: Relevant new datasets from HuggingFace

### 3. Storage Strategy
- **Category Path**: `datasets/{category}/{dataset_name}/{timestamp}/`
- **Metadata**: Rich metadata including category, confidence, query
- **Reusability**: Marked as reusable for future discovery

## Benefits

### For Researchers
- **Faster Discovery**: Find relevant datasets quickly by category
- **Quality Assurance**: Prioritize proven, high-quality datasets
- **Comprehensive Coverage**: Access both existing and new datasets
- **Reduced Redundancy**: Avoid downloading duplicate datasets

### For System Efficiency
- **Storage Optimization**: Reduce redundant data storage
- **Cost Reduction**: Minimize unnecessary downloads and storage
- **Scalability**: Organized structure scales with dataset growth
- **Maintainability**: Easy to browse, manage, and clean up

### For Team Collaboration
- **Shared Knowledge**: Build cumulative dataset library
- **Consistency**: Standardized organization across projects
- **Discoverability**: Easy for team members to find relevant data
- **Documentation**: Rich metadata for dataset understanding

## Usage Examples

### Basic Smart Discovery
```python
# Simple query processing
result = smart_dataset_discovery_tool("healthcare patient diagnosis")

# Extract recommendations
recommendations = json.loads(result)["recommendations"]
for rec in recommendations:
    print(f"{rec['type']} dataset: {rec['dataset']['name']}")
    print(f"Action: {rec['action']}")
    print(f"Priority: {rec['priority']}")
```

### Full Data Agent Workflow
```python
# Complete data collection with smart discovery
data_agent = DataAgent()
result = await data_agent.execute_data_collection(
    query="financial stock market prediction",
    session_id="finance_research_001"
)

# Check reusability metrics
summary = result["data_summary"]
print(f"Existing datasets reused: {summary['existing_datasets_reused']}")
print(f"New datasets added: {summary['new_datasets_added']}")
print(f"Category: {summary['category']}")
```

### Dataset Organization
```python
# Reorganize existing datasets (dry run first)
result = organize_dataset_categories_tool(dry_run=True)
plan = json.loads(result)

print(f"Datasets to reorganize: {plan['summary']['datasets_needing_reorganization']}")

# Execute reorganization
result = organize_dataset_categories_tool(dry_run=False)
```

## Configuration

### Environment Variables
```bash
# S3 Configuration
S3_BUCKET_NAME=ai-scientist-team-data
AWS_REGION=us-east-1

# API Keys (for dataset sources)
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
```

### Category Customization
You can extend or modify research categories in `data_tools.py`:

```python
RESEARCH_CATEGORIES = {
    "your-custom-category": [
        "keyword1", "keyword2", "phrase with spaces"
    ],
    # ... existing categories
}
```

## Testing

Run the comprehensive test suite:

```bash
cd python
python test/test_smart_data_discovery.py
```

This will test:
- Query categorization
- Existing dataset checking
- Smart discovery workflow
- Dataset organization
- Enhanced data agent functionality

## Migration

To migrate existing datasets to the new category structure:

1. **Analyze Current Structure**:
   ```python
   result = organize_dataset_categories_tool(dry_run=True)
   ```

2. **Review Migration Plan**:
   Check the reorganization plan and category assignments

3. **Execute Migration**:
   ```python
   result = organize_dataset_categories_tool(dry_run=False)
   ```

4. **Update Applications**:
   Update any hardcoded S3 paths to use the new category structure

## Best Practices

### For Queries
- Use descriptive, specific queries for better categorization
- Include domain-specific terms (e.g., "medical diagnosis" vs. "diagnosis")
- Combine multiple concepts when relevant

### For Storage
- Always pass the original query to `s3_storage_tool` for proper categorization
- Review category assignments and adjust if needed
- Use consistent naming conventions for datasets

### For Reusability
- Check existing datasets before starting new research
- Document dataset usage and quality assessments
- Share high-quality datasets with the team

## Future Enhancements

- **Machine Learning Categorization**: Use ML models for better query categorization
- **Semantic Search**: Enable semantic similarity search within categories
- **Usage Analytics**: Track dataset usage patterns for better recommendations
- **Auto-Tagging**: Automatically tag datasets with relevant keywords
- **Quality Scoring**: Implement automated quality assessment for datasets
- **Cross-Category Relationships**: Model relationships between categories

## Support

For questions or issues with the Smart Dataset Discovery System:

1. Check the test suite for usage examples
2. Review the tool documentation in `data_tools.py`
3. Examine the data models in `models/data.py`
4. Run the diagnostic tests to identify issues

The system is designed to be robust and provide fallbacks when external services are unavailable, ensuring reliable operation in all environments.