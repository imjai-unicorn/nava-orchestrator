// frontend/internal-dev-dashboard/src/components/DecisionViewer.jsx
import React, { useState, useEffect } from 'react';

const DecisionViewer = () => {
  const [decisions, setDecisions] = useState([]);
  const [selectedDecision, setSelectedDecision] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchDecisions = async () => {
      try {
        const response = await fetch('http://localhost:8005/api/admin/decisions');
        const data = await response.json();
        setDecisions(data.decisions || []);
      } catch (error) {
        console.error('Failed to fetch decisions:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchDecisions();
    const interval = setInterval(fetchDecisions, 10000); // Update every 10 seconds

    return () => clearInterval(interval);
  }, []);

  const formatTime = (timestamp) => {
    return new Date(timestamp).toLocaleTimeString();
  };

  const getModelColor = (model) => {
    switch (model) {
      case 'gpt': return 'bg-blue-100 text-blue-800';
      case 'claude': return 'bg-purple-100 text-purple-800';
      case 'gemini': return 'bg-green-100 text-green-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  if (loading) {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex justify-center items-center h-32">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h2 className="text-xl font-bold text-gray-800 mb-4">AI Decision Transparency</h2>
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Decision List */}
        <div>
          <h3 className="text-lg font-semibold mb-3">Recent Decisions</h3>
          <div className="space-y-2 max-h-96 overflow-y-auto">
            {decisions.map((decision, index) => (
              <div
                key={index}
                className={`p-3 border rounded-lg cursor-pointer transition-colors ${
                  selectedDecision?.id === decision.id 
                    ? 'bg-blue-50 border-blue-300' 
                    : 'hover:bg-gray-50'
                }`}
                onClick={() => setSelectedDecision(decision)}
              >
                <div className="flex justify-between items-center mb-1">
                  <span className={`px-2 py-1 rounded text-xs font-medium ${getModelColor(decision.selected_model)}`}>
                    {decision.selected_model?.toUpperCase()}
                  </span>
                  <span className="text-xs text-gray-500">{formatTime(decision.timestamp)}</span>
                </div>
                <div className="text-sm text-gray-700 truncate">
                  {decision.user_query || 'Query not available'}
                </div>
                <div className="text-xs text-gray-500 mt-1">
                  Confidence: {(decision.confidence * 100).toFixed(1)}%
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Decision Details */}
        <div>
          <h3 className="text-lg font-semibold mb-3">Decision Details</h3>
          {selectedDecision ? (
            <div className="border rounded-lg p-4 space-y-4">
              <div>
                <h4 className="font-medium text-gray-800 mb-2">Query</h4>
                <p className="text-sm text-gray-600 bg-gray-50 p-2 rounded">
                  {selectedDecision.user_query}
                </p>
              </div>

              <div>
                <h4 className="font-medium text-gray-800 mb-2">Selected Model</h4>
                <div className="flex items-center space-x-2">
                  <span className={`px-3 py-1 rounded text-sm font-medium ${getModelColor(selectedDecision.selected_model)}`}>
                    {selectedDecision.selected_model?.toUpperCase()}
                  </span>
                  <span className="text-sm text-gray-600">
                    Confidence: {(selectedDecision.confidence * 100).toFixed(1)}%
                  </span>
                </div>
              </div>

              <div>
                <h4 className="font-medium text-gray-800 mb-2">Reasoning</h4>
                <p className="text-sm text-gray-600">
                  {selectedDecision.reasoning || 'Model selected based on capability match and performance history.'}
                </p>
              </div>

              <div>
                <h4 className="font-medium text-gray-800 mb-2">Pattern Detected</h4>
                <span className="inline-block bg-indigo-100 text-indigo-800 px-2 py-1 rounded text-xs">
                  {selectedDecision.pattern || 'conversation'}
                </span>
              </div>

              <div>
                <h4 className="font-medium text-gray-800 mb-2">Alternative Scores</h4>
                <div className="space-y-1">
                  {selectedDecision.alternatives?.map((alt, idx) => (
                    <div key={idx} className="flex justify-between text-sm">
                      <span className="capitalize">{alt.model}</span>
                      <span className="text-gray-600">{(alt.score * 100).toFixed(1)}%</span>
                    </div>
                  )) || (
                    <div className="text-sm text-gray-500">No alternative scores available</div>
                  )}
                </div>
              </div>

              <div>
                <h4 className="font-medium text-gray-800 mb-2">Performance</h4>
                <div className="grid grid-cols-2 gap-2 text-sm">
                  <div>Response Time: {selectedDecision.response_time || 'N/A'}ms</div>
                  <div>Processing Time: {selectedDecision.processing_time || 'N/A'}ms</div>
                </div>
              </div>
            </div>
          ) : (
            <div className="border rounded-lg p-4 text-center text-gray-500">
              Select a decision to view details
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default DecisionViewer;