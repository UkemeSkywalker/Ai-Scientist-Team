# Technology Stack & Build System

## Frontend Stack

- **Framework**: Next.js 14 with TypeScript
- **Styling**: TailwindCSS with PostCSS
- **UI Components**: React 18 with custom components
- **Type Safety**: Full TypeScript coverage with strict configuration
- **Real-time Updates**: Strands SDK streaming API integration

## Backend Stack

- **Language**: Python 3.9+
- **Agent Framework**: Strands SDK for multi-agent orchestration
- **Data Models**: Pydantic v2 for type validation and serialization
- **Async Framework**: asyncio for concurrent agent execution
- **Logging**: structlog + Strands built-in observability
- **Environment**: python-dotenv for configuration management

## Key Dependencies

### Frontend

- Next.js, React, TypeScript
- TailwindCSS, PostCSS, Autoprefixer
- ESLint with Next.js configuration
- Strands SDK client libraries for real-time agent communication

### Backend

- **strands-agents** (Primary agent framework)
- boto3 (AWS SDK), pydantic, python-dotenv
- pandas, numpy (data processing)
- requests, httpx (HTTP clients)
- structlog (logging)
- pytest, pytest-asyncio (testing)
- black, flake8, mypy (code quality)
- **strands-tools** (Community tools package)

## Common Commands

### Frontend Development

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Start production server
npm start

# Run linting
npm run lint
```

### Backend Development

```bash
# Set up Python environment
cd python
python -m venv venv
source venv/bin/activate  # macOS/Linux

# Install dependencies (including Strands SDK)
pip install strands-agents
pip install -r requirements.txt

# Run Strands agent tests
python test_strands_orchestrator.py --query "test research"
python test_strands_research.py --query "AI ethics"
python test_shared_memory.py

# Run Strands-specific tests
pytest test_strands_agents.py
pytest test_a2a_communication.py
```

## Architecture Patterns

- **Strands Agent Framework**: All agents built using Strands SDK with agent-as-tools pattern
- **Agent-to-Agent (A2A) Communication**: Direct agent communication using Strands A2A protocol
- **Pydantic Models**: Strict type validation for all data structures
- **Strands Context Management**: Centralized state and conversation history through Strands SDK
- **Tool-First Development**: Build Strands tools before integrating into agents
- **Component-Based UI**: Reusable React components with Strands streaming integration
- **Cloud-Agnostic Design**: Multi-cloud support through Strands tools abstraction

## Strands SDK Knowledge Base

When implementing Strands SDK features or troubleshooting issues, use the Strands Agents knowledge base:

- **Search Tool**: Use `search_docs` with serverName "strands-agents" for Strands SDK documentation
- **Implementation Guidance**: Reference official Strands patterns and best practices
- **API Documentation**: Look up specific Strands SDK methods and configurations
- **Troubleshooting**: Find solutions for common Strands integration issues
