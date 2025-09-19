# AI Scientist Team

A multi-agent research system that automates research, analysis, experimentation, and visualization using Amazon Bedrock Strands agents orchestrated through Amazon Bedrock Agent Core.

## Overview

The AI Scientist Team consists of five specialized agents that work together to conduct comprehensive research:

1. **Research Agent** - Formulates hypotheses and searches literature
2. **Data Agent** - Finds and prepares datasets
3. **Experiment Agent** - Runs analyses and experiments
4. **Critic Agent** - Evaluates results and methodology
5. **Visualization Agent** - Creates charts and final reports

## Quick Start

### Prerequisites

- Node.js 18+ and npm
- Python 3.9+
- AWS Account (for production deployment)

### Setup

1. **Install Next.js dependencies:**
   ```bash
   npm install
   ```

2. **Set up Python environment:**
   ```bash
   cd python
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure environment:**
   ```bash
   cp python/.env.example python/.env
   # Edit python/.env with your AWS credentials
   ```

### Running the Application

1. **Start the Next.js frontend:**
   ```bash
   npm run dev
   ```
   Visit http://localhost:3000 to see the interface.

2. **Test the Python orchestrator:**
   ```bash
   cd python
   python test_orchestrator.py --query "Your research question"
   ```

## Testing Points

- **Frontend Test:** Submit a research query through the web interface and verify mock workflow progress displays
- **Backend Test:** Run the Python orchestrator test script and verify agent execution sequence
- **Integration Test:** Ensure both frontend and backend can communicate (future task)

## Project Structure

```
├── app/                    # Next.js frontend
│   ├── components/         # React components
│   ├── types/             # TypeScript type definitions
│   └── globals.css        # Global styles
├── python/                # Python backend
│   ├── src/
│   │   ├── core/          # Orchestrator and shared memory
│   │   └── models/        # Data models
│   └── test_orchestrator.py
└── README.md
```

## Current Status

✅ **Phase 1 Complete:** Minimal viable project structure with basic Next.js interface
- Working web interface with research query form
- Python project structure with mock agents
- Basic data models and type definitions
- Mock workflow orchestration
- Error handling and logging setup

## Next Steps

See `.kiro/specs/ai-scientist-team/tasks.md` for the complete implementation plan.

## Development

- **Frontend:** Next.js 14 with TypeScript and TailwindCSS
- **Backend:** Python with Pydantic models and structured logging
- **Future:** Amazon Bedrock integration for production deployment