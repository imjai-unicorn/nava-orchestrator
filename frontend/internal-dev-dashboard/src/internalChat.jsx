import React, { useState, useRef, useEffect } from 'react';

const InternalChat = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [selectedModel, setSelectedModel] = useState('auto');
  const [debugMode, setDebugMode] = useState(true);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [systemHealth, setSystemHealth] = useState(null);
  const fileInputRef = useRef(null);
  const messagesEndRef = useRef(null);

  // Auto scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim() && !uploadedFile) return;
    
    const userMessage = {
      id: Date.now(),
      content: input || '[File uploaded]',
      type: 'user',
      timestamp: new Date(),
      file: uploadedFile
    };
    
    setMessages(prev => [...prev, userMessage]);
    const messageToSend = input;
    const fileToSend = uploadedFile;
    setInput('');
    setUploadedFile(null);
    setLoading(true);

    try {
      const formData = new FormData();
      formData.append('message', messageToSend);
      formData.append('user_id', 'internal-dev-team');
      formData.append('preferred_model', selectedModel === 'auto' ? '' : selectedModel);
      formData.append('debug_mode', debugMode);
      
      if (fileToSend) {
        formData.append('file', fileToSend);
      }

      const response = await fetch('http://localhost:8005/api/chat', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      
      const aiMessage = {
        id: Date.now() + 1,
        content: data.response || 'No response received',
        type: 'ai',
        model: data.model_used || 'Unknown',
        reasoning: data.selection_reasoning,
        confidence: data.confidence,
        processing_time: data.processing_time,
        pattern_detected: data.pattern_detected,
        debug_info: data.debug_info,
        error_details: data.error_details,
        timestamp: new Date()
      };
      
      setMessages(prev => [...prev, aiMessage]);
      
    } catch (error) {
      console.error('API Error:', error);
      
      const errorMessage = {
        id: Date.now() + 2,
        content: `❌ Connection Error: ${error.message}\n\nTroubleshooting:\n1. Is NAVA backend running? → python main.py\n2. Check port 8005 availability\n3. Verify API endpoint: /api/chat\n4. Check CORS settings`,
        type: 'error',
        timestamp: new Date(),
        error_details: {
          name: error.name,
          message: error.message,
          stack: error.stack
        }
      };
      
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const testConnection = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8005/api/health');
      const data = await response.json();
      
      setSystemHealth(data);
      
      const healthMessage = {
        id: Date.now(),
        content: `✅ NAVA System Health:\n${JSON.stringify(data, null, 2)}`,
        type: 'system',
        timestamp: new Date()
      };
      
      setMessages(prev => [...prev, healthMessage]);
    } catch (error) {
      setSystemHealth({ status: 'error', error: error.message });
      
      const errorMessage = {
        id: Date.now(),
        content: `❌ Health Check Failed: ${error.message}\n\nNAVA backend is not running!\n\nStart backend:\ncd backend/services/01-core/nava-logic-controller\npython main.py`,
        type: 'error',
        timestamp: new Date()
      };
      
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      setUploadedFile(file);
    }
  };

  const clearChat = () => {
    setMessages([]);
    setSystemHealth(null);
  };

  const runQuickTest = (testType) => {
    const tests = {
      code: 'Write a Python function to calculate Fibonacci sequence with error handling',
      teaching: 'Explain machine learning concepts: supervised vs unsupervised learning',
      creative: 'Help me brainstorm innovative features for a mobile productivity app',
      analysis: 'Analyze the pros and cons of microservices architecture vs monolithic',
      debug: 'Debug this code: def fibonacci(n): return fibonacci(n-1) + fibonacci(n-2)',
      business: 'Create a business plan outline for an AI-powered startup'
    };
    setInput(tests[testType]);
  };

  return (
    <div style={{ 
      background: 'white', 
      padding: '24px', 
      borderRadius: '12px', 
      boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
      height: '90vh',
      display: 'flex',
      flexDirection: 'column'
    }}>
      {/* Header */}
      <div style={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center', 
        marginBottom: '16px',
        paddingBottom: '16px',
        borderBottom: '2px solid #e5e7eb'
      }}>
        <div>
          <h2 style={{ fontSize: '24px', fontWeight: 'bold', margin: '0' }}>
            💬 NAVA Internal Team Chat
          </h2>
          <p style={{ fontSize: '14px', color: '#6b7280', margin: '4px 0 0 0' }}>
            Development & Testing Interface
          </p>
        </div>
        
        {/* System Health Indicator */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{
            padding: '4px 8px',
            borderRadius: '12px',
            background: systemHealth?.status === 'healthy' ? '#dcfce7' : 
                       systemHealth?.status === 'error' ? '#fee2e2' : '#f3f4f6',
            color: systemHealth?.status === 'healthy' ? '#166534' : 
                   systemHealth?.status === 'error' ? '#dc2626' : '#374151',
            fontSize: '12px',
            fontWeight: 'bold'
          }}>
            {systemHealth?.status === 'healthy' ? '🟢 ONLINE' : 
             systemHealth?.status === 'error' ? '🔴 OFFLINE' : '⚪ UNKNOWN'}
          </div>
        </div>
      </div>

      {/* Controls */}
      <div style={{ 
        display: 'flex', 
        gap: '12px', 
        alignItems: 'center', 
        marginBottom: '16px',
        flexWrap: 'wrap'
      }}>
        <select
          value={selectedModel}
          onChange={(e) => setSelectedModel(e.target.value)}
          style={{ 
            padding: '8px 12px', 
            border: '1px solid #d1d5db', 
            borderRadius: '6px', 
            fontSize: '14px',
            background: 'white'
          }}
        >
          <option value="auto">🤖 NAVA Auto Select</option>
          <option value="gpt">🔵 Force GPT-4</option>
          <option value="claude">🟣 Force Claude</option>
          <option value="gemini">🟢 Force Gemini</option>
        </select>
        
        <label style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '14px' }}>
          <input
            type="checkbox"
            checked={debugMode}
            onChange={(e) => setDebugMode(e.target.checked)}
          />
          🔍 Debug Mode
        </label>
        
        <button
          onClick={testConnection}
          style={{
            padding: '8px 16px',
            background: '#10b981',
            color: 'white',
            border: 'none',
            borderRadius: '6px',
            fontSize: '14px',
            cursor: 'pointer',
            fontWeight: '500'
          }}
        >
          🔋 Health Check
        </button>
        
        <button
          onClick={clearChat}
          style={{
            padding: '8px 16px',
            background: '#ef4444',
            color: 'white',
            border: 'none',
            borderRadius: '6px',
            fontSize: '14px',
            cursor: 'pointer',
            fontWeight: '500'
          }}
        >
          🗑️ Clear Chat
        </button>
      </div>

      {/* Quick Test Buttons */}
      <div style={{ 
        marginBottom: '16px', 
        display: 'flex', 
        flexWrap: 'wrap', 
        gap: '8px' 
      }}>
        <button onClick={() => runQuickTest('code')} style={quickButtonStyle}>
          🔧 Code Gen
        </button>
        <button onClick={() => runQuickTest('teaching')} style={quickButtonStyle}>
          🧠 Teaching
        </button>
        <button onClick={() => runQuickTest('creative')} style={quickButtonStyle}>
          💡 Creative
        </button>
        <button onClick={() => runQuickTest('analysis')} style={quickButtonStyle}>
          📊 Analysis
        </button>
        <button onClick={() => runQuickTest('debug')} style={quickButtonStyle}>
          🐛 Debug
        </button>
        <button onClick={() => runQuickTest('business')} style={quickButtonStyle}>
          💼 Business
        </button>
      </div>
      
      {/* Messages - ขนาดใหญ่ขึ้น */}
      <div style={{ 
        flex: 1,
        overflowY: 'auto', 
        border: '2px solid #e5e7eb', 
        borderRadius: '12px', 
        padding: '20px', 
        marginBottom: '16px',
        background: '#f9fafb',
        minHeight: '400px'
      }}>
        {messages.length === 0 && (
          <div style={{ textAlign: 'center', color: '#6b7280', padding: '48px' }}>
            <div style={{ fontSize: '48px', marginBottom: '16px' }}>🔧</div>
            <h3 style={{ margin: '0 0 8px 0' }}>Internal Development Chat</h3>
            <p style={{ fontSize: '16px', margin: '0 0 12px 0' }}>
              Test NAVA logic, AI selection, and API integration
            </p>
            <p style={{ fontSize: '14px', margin: '0', opacity: 0.7 }}>
              Click "Health Check" to verify NAVA connection
            </p>
          </div>
        )}
        
        {messages.map((msg) => (
          <div key={msg.id} style={{ marginBottom: '20px' }}>
            <div style={{
              padding: '16px',
              borderRadius: '12px',
              background: 
                msg.type === 'user' ? '#3b82f6' : 
                msg.type === 'error' ? '#ef4444' :
                msg.type === 'system' ? '#10b981' : '#ffffff',
              color: 
                msg.type === 'user' || msg.type === 'error' || msg.type === 'system' ? 'white' : '#374151',
              maxWidth: msg.type === 'system' ? '100%' : '85%',
              marginLeft: msg.type === 'user' ? 'auto' : '0',
              border: msg.type === 'ai' ? '1px solid #e5e7eb' : 'none',
              fontSize: '15px',
              lineHeight: '1.5'
            }}>
              <div style={{ whiteSpace: 'pre-wrap' }}>{msg.content}</div>
              
              {/* File indicator */}
              {msg.file && (
                <div style={{ 
                  marginTop: '8px', 
                  fontSize: '12px', 
                  opacity: 0.8 
                }}>
                  📎 {msg.file.name} ({(msg.file.size / 1024).toFixed(1)} KB)
                </div>
              )}
              
              {/* AI Response Details */}
              {msg.type === 'ai' && debugMode && (
                <div style={{ 
                  marginTop: '12px', 
                  paddingTop: '12px', 
                  borderTop: '1px solid #e5e7eb',
                  fontSize: '13px',
                  background: '#f8fafc',
                  padding: '12px',
                  borderRadius: '8px',
                  color: '#374151'
                }}>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
                    <div><strong>🤖 Model:</strong> {msg.model}</div>
                    {msg.confidence && (
                      <div><strong>📊 Confidence:</strong> {(msg.confidence * 100).toFixed(1)}%</div>
                    )}
                    {msg.processing_time && (
                      <div><strong>⏱️ Time:</strong> {msg.processing_time}ms</div>
                    )}
                    {msg.pattern_detected && (
                      <div><strong>🎯 Pattern:</strong> {msg.pattern_detected}</div>
                    )}
                  </div>
                  {msg.reasoning && (
                    <div style={{ marginTop: '8px' }}>
                      <strong>🧠 Selection Reasoning:</strong> {msg.reasoning}
                    </div>
                  )}
                  {msg.debug_info && (
                    <div style={{ marginTop: '8px' }}>
                      <strong>🔍 Debug Info:</strong>
                      <pre style={{ fontSize: '11px', margin: '4px 0' }}>
                        {JSON.stringify(msg.debug_info, null, 2)}
                      </pre>
                    </div>
                  )}
                </div>
              )}

              {/* Error Details */}
              {msg.type === 'error' && msg.error_details && debugMode && (
                <div style={{ 
                  marginTop: '12px', 
                  paddingTop: '12px', 
                  borderTop: '1px solid rgba(255,255,255,0.2)',
                  fontSize: '12px',
                  opacity: 0.9
                }}>
                  <div><strong>Error Type:</strong> {msg.error_details.name}</div>
                  <div><strong>Message:</strong> {msg.error_details.message}</div>
                </div>
              )}
            </div>
            
            <div style={{ 
              fontSize: '12px', 
              color: '#6b7280', 
              marginTop: '6px', 
              textAlign: msg.type === 'user' ? 'right' : 'left' 
            }}>
              {msg.timestamp.toLocaleTimeString()}
            </div>
          </div>
        ))}
        
        {loading && (
          <div style={{ textAlign: 'center', padding: '20px' }}>
            <div style={{ 
              color: '#6b7280', 
              fontSize: '16px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '8px'
            }}>
              <div style={{ 
                width: '20px', 
                height: '20px', 
                border: '2px solid #e5e7eb', 
                borderTop: '2px solid #3b82f6', 
                borderRadius: '50%', 
                animation: 'spin 1s linear infinite' 
              }}></div>
              🤖 NAVA is processing...
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* File Upload Area */}
      {uploadedFile && (
        <div style={{
          background: '#f0f9ff',
          border: '1px solid #0ea5e9',
          borderRadius: '8px',
          padding: '12px',
          marginBottom: '16px',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center'
        }}>
          <div style={{ fontSize: '14px' }}>
            📎 <strong>{uploadedFile.name}</strong> ({(uploadedFile.size / 1024).toFixed(1)} KB)
          </div>
          <button
            onClick={() => setUploadedFile(null)}
            style={{
              background: '#ef4444',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              padding: '4px 8px',
              cursor: 'pointer',
              fontSize: '12px'
            }}
          >
            ✕
          </button>
        </div>
      )}

      {/* Input Area */}
      <div style={{ display: 'flex', gap: '12px', alignItems: 'flex-end' }}>
        <div style={{ flex: 1 }}>
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                if (!loading) sendMessage();
              }
            }}
            placeholder="Test NAVA: ask questions, request code, test AI selection... (Shift+Enter for new line)"
            disabled={loading}
            rows={3}
            style={{
              width: '100%',
              padding: '12px',
              border: '2px solid #d1d5db',
              borderRadius: '8px',
              fontSize: '14px',
              fontFamily: 'inherit',
              resize: 'vertical',
              minHeight: '60px'
            }}
          />
        </div>
        
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileUpload}
            style={{ display: 'none' }}
            accept=".txt,.pdf,.doc,.docx,.json,.csv,.py,.js,.jsx,.ts,.tsx"
          />
          
          <button
            onClick={() => fileInputRef.current?.click()}
            style={{
              padding: '12px',
              background: '#6b7280',
              color: 'white',
              border: 'none',
              borderRadius: '8px',
              cursor: 'pointer',
              fontSize: '16px'
            }}
          >
            📎
          </button>
          
          <button
            onClick={sendMessage}
            disabled={loading || (!input.trim() && !uploadedFile)}
            style={{
              padding: '12px 20px',
              background: loading ? '#9ca3af' : '#3b82f6',
              color: 'white',
              border: 'none',
              borderRadius: '8px',
              cursor: loading ? 'not-allowed' : 'pointer',
              fontSize: '14px',
              fontWeight: '500',
              minWidth: '80px'
            }}
          >
            {loading ? '⏳' : '📤 Send'}
          </button>
        </div>
      </div>

      {/* CSS Animation */}
      <style>{`
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
};

const quickButtonStyle = {
  padding: '6px 12px',
  background: '#f3f4f6',
  border: '1px solid #d1d5db',
  borderRadius: '6px',
  fontSize: '12px',
  cursor: 'pointer',
  fontWeight: '500',
  transition: 'all 0.2s',
};

export default InternalChat;