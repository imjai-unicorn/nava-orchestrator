import React, { useState, useEffect } from 'react';
import './AgentSelector.css';

const AgentSelector = ({ onModelSelect, selectedModel, disabled = false }) => {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Available AI models with descriptions
  const modelInfo = {
    'gpt-4o-mini': {
      name: 'GPT-4O Mini',
      provider: 'OpenAI',
      description: 'Fast and efficient for general conversations',
      speed: 'Fast',
      cost: 'Low',
      capabilities: ['Conversation', 'Code', 'Analysis']
    },
    'gpt-4o': {
      name: 'GPT-4O',
      provider: 'OpenAI',
      description: 'Advanced model for complex tasks',
      speed: 'Medium',
      cost: 'Medium',
      capabilities: ['Conversation', 'Code', 'Analysis', 'Complex Reasoning']
    },
    'claude-3-5-sonnet-20241022': {
      name: 'Claude 3.5 Sonnet',
      provider: 'Anthropic',
      description: 'Excellent for deep analysis and reasoning',
      speed: 'Medium',
      cost: 'Medium',
      capabilities: ['Deep Analysis', 'Reasoning', 'Code Review']
    },
    'claude-3-haiku-20240307': {
      name: 'Claude 3 Haiku',
      provider: 'Anthropic',
      description: 'Quick responses for simple tasks',
      speed: 'Fast',
      cost: 'Low',
      capabilities: ['Conversation', 'Quick Q&A']
    },
    'gemini-1.5-flash': {
      name: 'Gemini 1.5 Flash',
      provider: 'Google',
      description: 'Multimodal capabilities with fast processing',
      speed: 'Fast',
      cost: 'Low',
      capabilities: ['Multimodal', 'Research', 'Document Processing']
    }
  };

  useEffect(() => {
    fetchAvailableModels();
  }, []);

  const fetchAvailableModels = async () => {
    try {
      setLoading(true);
      // In a real app, this would fetch from API
      const response = await fetch('/api/models');
      if (response.ok) {
        const data = await response.json();
        setModels(data.models || Object.keys(modelInfo));
      } else {
        // Fallback to default models
        setModels(Object.keys(modelInfo));
      }
    } catch (err) {
      console.error('Error fetching models:', err);
      setError('Failed to load models');
      // Fallback to default models
      setModels(Object.keys(modelInfo));
    } finally {
      setLoading(false);
    }
  };

  const handleModelSelect = (modelId) => {
    if (!disabled) {
      onModelSelect(modelId);
    }
  };

  const getSpeedColor = (speed) => {
    switch (speed) {
      case 'Fast': return 'text-green-600';
      case 'Medium': return 'text-yellow-600';
      case 'Slow': return 'text-red-600';
      default: return 'text-gray-600';
    }
  };

  const getCostColor = (cost) => {
    switch (cost) {
      case 'Low': return 'text-green-600';
      case 'Medium': return 'text-yellow-600';
      case 'High': return 'text-red-600';
      default: return 'text-gray-600';
    }
  };

  if (loading) {
    return (
      <div className="agent-selector loading">
        <div className="loading-spinner"></div>
        <p>Loading AI models...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="agent-selector error">
        <p className="error-message">{error}</p>
        <button onClick={fetchAvailableModels} className="retry-button">
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className="agent-selector">
      <h3>Choose AI Model</h3>
      <div className="model-grid">
        {models.map((modelId) => {
          const info = modelInfo[modelId];
          if (!info) return null;

          return (
            <div
              key={modelId}
              className={`model-card ${selectedModel === modelId ? 'selected' : ''} ${disabled ? 'disabled' : ''}`}
              onClick={() => handleModelSelect(modelId)}
            >
              <div className="model-header">
                <h4>{info.name}</h4>
                <span className="provider">{info.provider}</span>
              </div>
              
              <p className="description">{info.description}</p>
              
              <div className="model-stats">
                <div className="stat">
                  <span className="label">Speed:</span>
                  <span className={`value ${getSpeedColor(info.speed)}`}>{info.speed}</span>
                </div>
                <div className="stat">
                  <span className="label">Cost:</span>
                  <span className={`value ${getCostColor(info.cost)}`}>{info.cost}</span>
                </div>
              </div>
              
              <div className="capabilities">
                <span className="label">Capabilities:</span>
                <div className="capability-tags">
                  {info.capabilities.map((cap, idx) => (
                    <span key={idx} className="capability-tag">{cap}</span>
                  ))}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default AgentSelector;