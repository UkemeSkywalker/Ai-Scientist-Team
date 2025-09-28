# AI Scientist Team - Project Overview

## What This Project Does

The **AI Scientist Team** is a sophisticated multi-agent research automation system that orchestrates five specialized AI agents to conduct end-to-end research workflows. The project automates the entire research lifecycle from hypothesis generation to final visualization.

## Core Architecture

### Multi-Agent System Design
The system uses **Amazon Bedrock Strands SDK** to orchestrate five specialized agents:

1. **Research Agent** - Literature search, hypothesis generation, research synthesis
2. **Data Agent** - Dataset discovery, cleaning, and storage (Kaggle, HuggingFace, AWS Open Data)
3. **Experiment Agent** - ML experiments and statistical analysis using SageMaker
4. **Critic Agent** - Quality assurance, bias detection, methodology validation
5. **Visualization Agent** - Chart generation and report creation

### Technology Stack
- **Frontend**: Next.js 14 + TypeScript + TailwindCSS
- **Backend**: Python with Strands SDK for agent orchestration
- **Data Models**: Pydantic v2 for type validation
- **Storage**: Local file-based shared memory (transitioning to Bedrock)
- **Cloud Services**: AWS S3, SageMaker, QuickSight integration
- **APIs**: arXiv, PubMed, Kaggle, HuggingFace Datasets

## Current Implementation Status

### ✅ Completed (Phase 1)
- Next.js frontend with research query interface
- Python project structure with Pydantic models
- Local shared memory system with JSON persistence
- Mock agent orchestration workflow
- Research Agent with external API tools (arXiv, PubMed)
- Data Agent with dataset discovery capabilities
- Experiment Agent with SageMaker integration
- Comprehensive testing framework

### 🔄 In Progress
- Strands SDK integration (partially implemented)
- Agent-to-Agent (A2A) communication
- Real-time progress tracking in frontend
- Enhanced error handling and observability

### 📋 Planned (Phases 2-4)
- Critic Agent implementation
- Visualization Agent with chart generation
- Full Strands SDK orchestration
- Production deployment with cloud-agnostic support

## Key Features

### Transparent Research Process
- Each agent's output is displayed to users before the next agent executes
- Real-time progress tracking with conversation history
- Step-by-step workflow visualization

### Advanced Agent Collaboration
- Shared memory context management
- Agent-to-Agent (A2A) communication protocol
- Peer validation and cross-agent quality checks

### Research Automation
- Automated literature search across multiple databases
- Hypothesis generation and testing
- Dataset discovery and preprocessing
- Statistical analysis and ML experimentation
- Critical evaluation and bias detection
- Automated report generation with citations

## Project Structure

```
├── app/                    # Next.js frontend
│   ├── components/         # React components
│   ├── types/             # TypeScript type definitions
│   └── globals.css        # Global styles
├── python/                # Python backend
│   ├── src/
│   │   ├── agents/        # Strands agent implementations
│   │   ├── tools/         # Strands tool implementations
│   │   ├── core/          # Orchestrator and shared memory
│   │   ├── models/        # Pydantic data models
│   │   └── integrations/  # Cloud service integrations
│   ├── test/              # Comprehensive test suite
│   └── data/              # Runtime data storage
├── .kiro/                 # Project specifications
│   ├── specs/             # Requirements, design, tasks
│   └── steering/          # Product vision, tech stack
└── README.md
```

## Current Capabilities

The system can currently:
- Accept research queries through a web interface
- Search academic literature (arXiv, PubMed)
- Generate testable hypotheses
- Discover and process datasets from multiple sources
- Run ML experiments on SageMaker
- Store results in structured shared memory
- Generate bibliographies in multiple formats (APA, MLA)
- Export research outputs to organized file structures

## How to Run

### Frontend
```bash
npm install
npm run dev
# Visit http://localhost:3000
```

### Backend Testing
```bash
cd python
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python test/test_strands_research.py --query "AI ethics"
```

## Project Vision

This is an ambitious research automation platform that aims to democratize scientific research by making the entire research workflow accessible through AI agent collaboration. The system provides transparency at every step, allowing users to understand and trust the research process while leveraging the power of specialized AI agents.

## Documentation

The `.kiro` directory contains comprehensive project specifications:
- **Requirements**: Detailed user stories and acceptance criteria
- **Design**: Strands SDK architecture and agent patterns  
- **Tasks**: Phase-by-phase implementation plan with testing points
- **Steering**: Product vision, tech stack, and project organization

## Next Steps

The project is currently in Phase 2, focusing on completing the remaining Strands agents (Critic and Visualization) and implementing advanced features like Agent-to-Agent communication and real-time frontend integration.