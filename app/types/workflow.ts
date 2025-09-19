export type AgentType = 'research' | 'data' | 'experiment' | 'critic' | 'visualization'

export type AgentStatusType = 'pending' | 'running' | 'completed' | 'failed'

export type WorkflowStatusType = 'idle' | 'running' | 'completed' | 'failed'

export interface AgentStatus {
  status: AgentStatusType
  progress: number
  results: any | null
  error?: string
  startTime?: Date
  endTime?: Date
}

export interface WorkflowState {
  sessionId: string
  query: string
  status: WorkflowStatusType
  currentAgent: AgentType
  agents: Record<AgentType, AgentStatus>
  startTime: Date
  endTime?: Date
  estimatedDuration: number
  error?: string
}

export interface ResearchFindings {
  hypotheses: string[]
  literatureCount: number
  keyFindings: string
}

export interface DataContext {
  datasetsFound: number
  totalSamples: number
  qualityScore: number
  sources: string[]
}

export interface ExperimentResults {
  experimentsRun: number
  bestAccuracy: number
  statisticalSignificance: number
  modelType: string
}

export interface CriticalEvaluation {
  overallConfidence: number
  limitations: string[]
  recommendations: string[]
}

export interface VisualizationResults {
  chartsGenerated: number
  dashboardUrl: string
  reportGenerated: boolean
  exportFormats: string[]
}