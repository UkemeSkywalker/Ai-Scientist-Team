"use client";

import { useState } from "react";
import ResearchForm from "./components/ResearchForm";
import WorkflowProgress from "./components/WorkflowProgress";
import { WorkflowState, AgentStatus } from "./types/workflow";

export default function Home() {
  const [workflowState, setWorkflowState] = useState<WorkflowState | null>(
    null
  );

  const handleResearchSubmit = async (query: string) => {
    // Initialize mock workflow
    const initialState: WorkflowState = {
      sessionId: `session_${Date.now()}`,
      query,
      status: "running",
      currentAgent: "research",
      agents: {
        research: { status: "running", progress: 0, results: null },
        data: { status: "pending", progress: 0, results: null },
        experiment: { status: "pending", progress: 0, results: null },
        critic: { status: "pending", progress: 0, results: null },
        visualization: { status: "pending", progress: 0, results: null },
      },
      startTime: new Date(),
      estimatedDuration: 300, // 5 minutes mock duration
    };

    setWorkflowState(initialState);

    // Simulate agent execution with mock responses
    await simulateWorkflow(initialState, setWorkflowState);
  };

  return (
    <div className="space-y-8">
      <div className="text-center">
        <h2 className="text-3xl font-bold tracking-tight text-gray-900 sm:text-4xl">
          Automated Research Assistant
        </h2>
        <p className="mt-4 text-lg text-gray-600">
          Submit your research question and watch our AI agents collaborate to
          provide comprehensive analysis
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div>
          {workflowState && <WorkflowProgress workflowState={workflowState} />}
        </div>

        <div>
          <ResearchForm
            onSubmit={handleResearchSubmit}
            isLoading={workflowState?.status === "running"}
          />
        </div>
      </div>
    </div>
  );
}

// Mock workflow simulation
async function simulateWorkflow(
  initialState: WorkflowState,
  setState: React.Dispatch<React.SetStateAction<WorkflowState | null>>
) {
  const agents = [
    "research",
    "data",
    "experiment",
    "critic",
    "visualization",
  ] as const;

  for (let i = 0; i < agents.length; i++) {
    const currentAgent = agents[i];
    const nextAgent = agents[i + 1];

    // Update current agent to running
    setState((prev: WorkflowState | null) =>
      prev
        ? {
            ...prev,
            currentAgent,
            agents: {
              ...prev.agents,
              [currentAgent]: {
                ...prev.agents[currentAgent],
                status: "running",
              },
            },
          }
        : initialState
    );

    // Simulate progress
    for (let progress = 0; progress <= 100; progress += 20) {
      await new Promise((resolve) => setTimeout(resolve, 500));
      setState((prev: WorkflowState | null) =>
        prev
          ? {
              ...prev,
              agents: {
                ...prev.agents,
                [currentAgent]: { ...prev.agents[currentAgent], progress },
              },
            }
          : initialState
      );
    }

    // Complete current agent with mock results
    const mockResults = getMockResults(currentAgent);
    setState((prev: WorkflowState | null) =>
      prev
        ? {
            ...prev,
            agents: {
              ...prev.agents,
              [currentAgent]: {
                status: "completed",
                progress: 100,
                results: mockResults,
              },
            },
          }
        : initialState
    );

    // Set next agent to running if exists
    if (nextAgent) {
      setState((prev: WorkflowState | null) =>
        prev
          ? {
              ...prev,
              currentAgent: nextAgent,
              agents: {
                ...prev.agents,
                [nextAgent]: { ...prev.agents[nextAgent], status: "running" },
              },
            }
          : initialState
      );
    }
  }

  // Mark workflow as completed
  setState((prev: WorkflowState | null) =>
    prev
      ? {
          ...prev,
          status: "completed",
          currentAgent: "visualization",
          endTime: new Date(),
        }
      : initialState
  );
}

function getMockResults(agent: string) {
  const mockResults = {
    research: {
      hypotheses: [
        "Machine learning bias can be reduced through diverse training data",
        "Algorithmic fairness metrics can detect discriminatory patterns",
      ],
      literatureCount: 47,
      keyFindings: "Found significant research on bias mitigation techniques",
    },
    data: {
      datasetsFound: 3,
      totalSamples: 125000,
      qualityScore: 0.87,
      sources: ["Kaggle", "HuggingFace", "AWS Open Data"],
    },
    experiment: {
      experimentsRun: 5,
      bestAccuracy: 0.94,
      statisticalSignificance: 0.001,
      modelType: "Random Forest Classifier",
    },
    critic: {
      overallConfidence: 0.82,
      limitations: [
        "Limited demographic diversity",
        "Potential selection bias",
      ],
      recommendations: [
        "Expand dataset diversity",
        "Cross-validate with external data",
      ],
    },
    visualization: {
      chartsGenerated: 8,
      dashboardUrl: "/mock-dashboard",
      reportGenerated: true,
      exportFormats: ["PDF", "HTML", "Interactive"],
    },
  };

  return mockResults[agent as keyof typeof mockResults];
}
