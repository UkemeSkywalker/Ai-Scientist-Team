'use client'

import { WorkflowState, AgentType } from '../types/workflow'

interface WorkflowProgressProps {
  workflowState: WorkflowState
}

export default function WorkflowProgress({ workflowState }: WorkflowProgressProps) {
  const agentInfo = {
    research: {
      name: 'Research Agent',
      description: 'Formulating hypotheses and searching literature',
      icon: '🔍'
    },
    data: {
      name: 'Data Agent',
      description: 'Finding and preparing datasets',
      icon: '📊'
    },
    experiment: {
      name: 'Experiment Agent',
      description: 'Running analyses and experiments',
      icon: '🧪'
    },
    critic: {
      name: 'Critic Agent',
      description: 'Evaluating results and methodology',
      icon: '🔬'
    },
    visualization: {
      name: 'Visualization Agent',
      description: 'Creating charts and final report',
      icon: '📈'
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'text-green-600 bg-green-100'
      case 'running': return 'text-blue-600 bg-blue-100'
      case 'failed': return 'text-red-600 bg-red-100'
      default: return 'text-gray-600 bg-gray-100'
    }
  }

  const formatDuration = (start: Date, end?: Date) => {
    const duration = (end || new Date()).getTime() - start.getTime()
    const seconds = Math.floor(duration / 1000)
    const minutes = Math.floor(seconds / 60)
    return minutes > 0 ? `${minutes}m ${seconds % 60}s` : `${seconds}s`
  }

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold text-gray-900">
          Research Progress
        </h3>
        <div className="flex items-center space-x-2">
          <span className={`px-2 py-1 text-xs font-medium rounded-full ${
            workflowState.status === 'completed' ? 'bg-green-100 text-green-800' :
            workflowState.status === 'running' ? 'bg-blue-100 text-blue-800' :
            workflowState.status === 'failed' ? 'bg-red-100 text-red-800' :
            'bg-gray-100 text-gray-800'
          }`}>
            {workflowState.status}
          </span>
          <span className="text-sm text-gray-500">
            {formatDuration(workflowState.startTime, workflowState.endTime)}
          </span>
        </div>
      </div>

      <div className="mb-4 p-3 bg-gray-50 rounded-md">
        <p className="text-sm text-gray-700">
          <strong>Query:</strong> {workflowState.query}
        </p>
      </div>

      <div className="space-y-4">
        {(Object.keys(agentInfo) as AgentType[]).map((agentType) => {
          const agent = workflowState.agents[agentType]
          const info = agentInfo[agentType]
          
          return (
            <div key={agentType} className="border rounded-lg p-4">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center space-x-3">
                  <span className="text-2xl">{info.icon}</span>
                  <div>
                    <h4 className="font-medium text-gray-900">{info.name}</h4>
                    <p className="text-sm text-gray-600">{info.description}</p>
                  </div>
                </div>
                <span className={`px-2 py-1 text-xs font-medium rounded-full ${getStatusColor(agent.status)}`}>
                  {agent.status}
                </span>
              </div>

              {agent.status === 'running' && (
                <div className="mb-3">
                  <div className="flex justify-between text-sm text-gray-600 mb-1">
                    <span>Progress</span>
                    <span>{agent.progress}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-blue-600 h-2 rounded-full transition-all duration-500"
                      style={{ width: `${agent.progress}%` }}
                    ></div>
                  </div>
                </div>
              )}

              {agent.results && (
                <div className="mt-3 p-3 bg-gray-50 rounded-md">
                  <h5 className="text-sm font-medium text-gray-900 mb-2">Results:</h5>
                  <div className="text-sm text-gray-700">
                    {agentType === 'research' && (
                      <div>
                        <p>• {agent.results.hypotheses?.length} hypotheses generated</p>
                        <p>• {agent.results.literatureCount} papers reviewed</p>
                        <p>• {agent.results.keyFindings}</p>
                      </div>
                    )}
                    {agentType === 'data' && (
                      <div>
                        <p>• {agent.results.datasetsFound} datasets found</p>
                        <p>• {agent.results.totalSamples?.toLocaleString()} total samples</p>
                        <p>• Quality score: {(agent.results.qualityScore * 100).toFixed(1)}%</p>
                      </div>
                    )}
                    {agentType === 'experiment' && (
                      <div>
                        <p>• {agent.results.experimentsRun} experiments completed</p>
                        <p>• Best accuracy: {(agent.results.bestAccuracy * 100).toFixed(1)}%</p>
                        <p>• Model: {agent.results.modelType}</p>
                      </div>
                    )}
                    {agentType === 'critic' && (
                      <div>
                        <p>• Confidence: {(agent.results.overallConfidence * 100).toFixed(1)}%</p>
                        <p>• {agent.results.limitations?.length} limitations identified</p>
                        <p>• {agent.results.recommendations?.length} recommendations provided</p>
                      </div>
                    )}
                    {agentType === 'visualization' && (
                      <div>
                        <p>• {agent.results.chartsGenerated} visualizations created</p>
                        <p>• Report generated: {agent.results.reportGenerated ? 'Yes' : 'No'}</p>
                        <p>• Export formats: {agent.results.exportFormats?.join(', ')}</p>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}