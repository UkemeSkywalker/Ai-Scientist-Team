'use client'

import { useState } from 'react'

interface ResearchFormProps {
  onSubmit: (query: string) => void
  isLoading: boolean
}

export default function ResearchForm({ onSubmit, isLoading }: ResearchFormProps) {
  const [query, setQuery] = useState('')

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (query.trim() && !isLoading) {
      onSubmit(query.trim())
    }
  }

  const exampleQueries = [
    "How can machine learning bias be reduced in healthcare applications?",
    "What are the most effective methods for detecting fake news using NLP?",
    "How does climate change affect agricultural productivity in developing countries?",
    "What are the latest advances in quantum computing for cryptography?"
  ]

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">
        Submit Research Query
      </h3>
      
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label htmlFor="query" className="block text-sm font-medium text-gray-700 mb-2">
            Research Question
          </label>
          <textarea
            id="query"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Enter your research question here..."
            className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 resize-none text-gray-900"
            rows={4}
            disabled={isLoading}
            required
          />
        </div>

        <button
          type="submit"
          disabled={!query.trim() || isLoading}
          className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {isLoading ? (
            <div className="flex items-center justify-center">
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
              Processing Research...
            </div>
          ) : (
            'Start Research'
          )}
        </button>
      </form>

      <div className="mt-6">
        <h4 className="text-sm font-medium text-gray-700 mb-3">Example Queries:</h4>
        <div className="space-y-2">
          {exampleQueries.map((example, index) => (
            <button
              key={index}
              onClick={() => !isLoading && setQuery(example)}
              disabled={isLoading}
              className="text-left w-full p-2 text-sm text-blue-600 hover:text-blue-800 hover:bg-blue-50 rounded border border-transparent hover:border-blue-200 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {example}
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}