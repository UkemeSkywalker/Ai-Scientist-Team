# Project Structure & Organization

## Root Directory Layout
```
├── app/                    # Next.js frontend application
├── python/                 # Python backend and orchestration
├── .kiro/                  # Kiro IDE configuration and specs
├── .next/                  # Next.js build artifacts (auto-generated)
├── node_modules/           # Frontend dependencies (auto-generated)
└── Configuration files     # package.json, tsconfig.json, etc.
```

## Frontend Structure (`app/`)
```
app/
├── components/             # Reusable React components
│   ├── ResearchForm.tsx   # Research query input form
│   ├── StrandsProgress.tsx # Strands agent progress visualization
│   ├── AgentConversation.tsx # Strands conversation history display
│   ├── A2AMonitor.tsx     # Agent-to-Agent communication monitor
│   └── StrandsObservability.tsx # Strands tracing and metrics display
├── types/                 # TypeScript type definitions
│   ├── strands.ts         # Strands agent and tool type definitions
│   ├── workflow.ts        # Workflow state type definitions
│   └── a2a.ts             # Agent-to-Agent communication types
├── hooks/                 # React hooks for Strands integration
│   ├── useStrandsStream.ts # Hook for Strands streaming API
│   └── useA2AMonitor.ts   # Hook for A2A communication monitoring
├── globals.css            # Global styles and Tailwind imports
├── layout.tsx             # Root layout component
└── page.tsx               # Main application page with Strands integration
```

## Backend Structure (`python/`)
```
python/
├── src/                   # Source code modules
│   ├── agents/           # Strands agent implementations
│   │   ├── orchestrator.py    # Strands orchestrator agent
│   │   ├── research_agent.py  # Research Strands agent with tools
│   │   ├── data_agent.py      # Data Strands agent with tools
│   │   ├── experiment_agent.py # Experiment Strands agent with tools
│   │   ├── critic_agent.py    # Critic Strands agent with tools
│   │   └── viz_agent.py       # Visualization Strands agent with tools
│   ├── tools/            # Strands tool implementations
│   │   ├── research_tools.py  # arXiv, PubMed, hypothesis tools
│   │   ├── data_tools.py      # Kaggle, HuggingFace, S3 tools
│   │   ├── experiment_tools.py # SageMaker, statistical analysis tools
│   │   ├── critic_tools.py    # Validation, bias detection tools
│   │   └── viz_tools.py       # Chart generation, report tools
│   ├── core/             # Core system components
│   │   ├── shared_memory.py   # Enhanced context management
│   │   ├── logger.py          # Structured logging + Strands observability
│   │   └── a2a_handler.py     # Agent-to-Agent communication handler
│   ├── models/           # Pydantic data models
│   │   ├── workflow.py        # Strands workflow state models
│   │   ├── research.py        # Research context models
│   │   ├── data.py            # Data processing models
│   │   ├── experiment.py      # Experiment result models
│   │   ├── critic.py          # Evaluation models
│   │   └── visualization.py   # Visualization models
│   └── integrations/     # Cloud service integrations
│       ├── strands_config.py  # Strands SDK configuration
│       ├── aws_tools.py       # AWS service tools
│       ├── gcp_tools.py       # GCP service tools (future)
│       └── azure_tools.py     # Azure service tools (future)
├── data/                 # Runtime data storage
│   ├── strands_context/       # Strands conversation history
│   ├── demo_sessions/         # Demo session data
│   └── test_sessions/         # Test session data
├── test_strands_*.py     # Strands agent test scripts
├── requirements.txt      # Python dependencies (including strands-agents)
└── .env.example          # Environment configuration with Strands settings
```

## Key Architectural Patterns

### Strands Agent Organization
- Each Strands agent has its implementation in `python/src/agents/`
- All agents use Strands SDK with agent-as-tools pattern
- Agent tools are organized in `python/src/tools/` by functionality
- Agent state is managed through Strands context management
- A2A communication handled via `python/src/core/a2a_handler.py`

### Type Safety
- Python: Pydantic models ensure runtime type validation for Strands tools
- TypeScript: Strict typing with interfaces matching Strands agent models
- Shared enums between frontend and backend for Strands agent states
- Strands SDK provides built-in type validation for agent interactions

### Data Flow
1. **Frontend** → User submits query via `ResearchForm`
2. **Strands Orchestrator** → Routes query to appropriate Strands agents as tools
3. **Strands Context** → Manages conversation history and agent state
4. **A2A Communication** → Enables direct agent-to-agent interactions
5. **Frontend** → Displays real-time progress via Strands streaming API

### File Naming Conventions
- **Python**: snake_case for files and functions
- **Strands Agents**: `*_agent.py` for agent implementations
- **Strands Tools**: `*_tools.py` for tool collections
- **TypeScript**: PascalCase for components, camelCase for functions
- **Models**: Singular names (workflow.py, not workflows.py)
- **Components**: Descriptive names with Strands prefix where applicable (StrandsProgress, A2AMonitor)

### Configuration Management
- Environment variables in `.env` files (Python + Strands SDK settings)
- Strands SDK configuration in `python/src/integrations/strands_config.py`
- Next.js configuration in `next.config.js`
- TypeScript configuration in `tsconfig.json`
- Tailwind configuration in `tailwind.config.js`
- A2A protocol configuration for agent communication

## Strands SDK Implementation Guidelines

When implementing Strands SDK features:

- **Documentation Lookup**: Use `search_docs` with serverName "strands-agents" for official documentation
- **Agent Patterns**: Follow Strands agent-as-tools pattern for all agent implementations
- **Tool Development**: Reference Strands tool creation guidelines and best practices
- **A2A Communication**: Use official Strands A2A protocol documentation for agent-to-agent features
- **Error Handling**: Leverage Strands SDK built-in error handling and retry mechanisms
- **Testing**: Use Strands testing framework for agent and tool validation