// frontend/internal-dev-dashboard/src/components/APITester.jsx
import React, { useState, useEffect } from 'react';

const APITester = () => {
  const [endpoint, setEndpoint] = useState('/api/chat');
  const [method, setMethod] = useState('POST');
  const [requestBody, setRequestBody] = useState('{\n  "message": "Hello, NAVA!",\n  "user_id": "test-user"\n}');
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [headers, setHeaders] = useState('{\n  "Content-Type": "application/json"\n}');

  const predefinedEndpoints = [
    { path: '/api/chat', method: 'POST', body: '{\n  "message": "Hello, NAVA!",\n  "user_id": "test-user"\n}' },
    { path: '/api/health', method: 'GET', body: '' },
    { path: '/api/admin/decisions', method: 'GET', body: '' },
    { path: '/api/admin/system-status', method: 'GET', body: '' },
    { path: '/api/admin/feature-flags', method: 'GET', body: '' }
  ];

  const sendRequest = async () => {
    setLoading(true);
    setResponse(null);

    const startTime = Date.now();
    
    try {
      let parsedHeaders = {};
      try {
        parsedHeaders = JSON.parse(headers);
      } catch (e) {
        parsedHeaders = { 'Content-Type': 'application/json' };
      }

      const requestOptions = {
        method: method,
        headers: parsedHeaders
      };

      if (method !== 'GET' && requestBody.trim()) {
        requestOptions.body = requestBody;
      }

      const res = await fetch(`http://localhost:8005${endpoint}`, requestOptions);
      const endTime = Date.now();
      
      let responseData;
      const contentType = res.headers.get('content-type');
      
      if (contentType && contentType.includes('application/json')) {
        responseData = await res.json();
      } else {
        responseData = await res.text();
      }

      setResponse({
        status: res.status,
        statusText: res.statusText,
        headers: Object.fromEntries(res.headers.entries()),
        data: responseData,
        time: endTime - startTime,
        success: res.ok
      });
    } catch (error) {
      const endTime = Date.now();
      setResponse({
        status: 0,
        statusText: 'Network Error',
        headers: {},
        data: { error: error.message },
        time: endTime - startTime,
        success: false
      });
    } finally {
      setLoading(false);
    }
  };

  const loadPredefinedEndpoint = (predefined) => {
    setEndpoint(predefined.path);
    setMethod(predefined.method);
    setRequestBody(predefined.body);
  };

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h2 className="text-xl font-bold text-gray-800 mb-4">API Testing Tools</h2>
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Request Panel */}
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Quick Load Predefined Endpoints
            </label>
            <div className="flex flex-wrap gap-2">
              {predefinedEndpoints.map((pred, idx) => (
                <button
                  key={idx}
                  onClick={() => loadPredefinedEndpoint(pred)}
                  className="px-3 py-1 bg-blue-100 text-blue-700 rounded text-sm hover:bg-blue-200 transition-colors"
                >
                  {pred.method} {pred.path}
                </button>
              ))}
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              HTTP Method
            </label>
            <select
              value={method}
              onChange={(e) => setMethod(e.target.value)}
              className="w-full p-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
            >
              <option value="GET">GET</option>
              <option value="POST">POST</option>
              <option value="PUT">PUT</option>
              <option value="DELETE">DELETE</option>
              <option value="PATCH">PATCH</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Endpoint
            </label>
            <div className="flex">
              <span className="inline-flex items-center px-3 text-sm text-gray-900 bg-gray-200 border border-r-0 border-gray-300 rounded-l-md">
                localhost:8005
              </span>
              <input
                type="text"
                value={endpoint}
                onChange={(e) => setEndpoint(e.target.value)}
                className="flex-1 p-2 border border-gray-300 rounded-r-md focus:ring-blue-500 focus:border-blue-500"
                placeholder="/api/endpoint"
              />
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Headers (JSON)
            </label>
            <textarea
              value={headers}
              onChange={(e) => setHeaders(e.target.value)}
              rows={3}
              className="w-full p-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500 font-mono text-sm"
              placeholder="Request headers in JSON format"
            />
          </div>

          {method !== 'GET' && (
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Request Body (JSON)
              </label>
              <textarea
                value={requestBody}
                onChange={(e) => setRequestBody(e.target.value)}
                rows={6}
                className="w-full p-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500 font-mono text-sm"
                placeholder="Request body in JSON format"
              />
            </div>
          )}

          <button
            onClick={sendRequest}
            disabled={loading}
            className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 focus:ring-4 focus:ring-blue-200 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {loading ? (
              <div className="flex items-center justify-center">
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                Sending...
              </div>
            ) : (
              'Send Request'
            )}
          </button>
        </div>

        {/* Response Panel */}
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-gray-800">Response</h3>
          
          {response ? (
            <div className="space-y-4">
              {/* Status and Time */}
              <div className="flex justify-between items-center p-3 bg-gray-50 rounded-md">
                <div className="flex items-center space-x-2">
                  <span className={`px-2 py-1 rounded text-sm font-medium ${
                    response.success ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                  }`}>
                    {response.status} {response.statusText}
                  </span>
                </div>
                <div className="text-sm text-gray-600">
                  {response.time}ms
                </div>
              </div>

              {/* Response Headers */}
              <div>
                <h4 className="font-medium text-gray-800 mb-2">Response Headers</h4>
                <pre className="bg-gray-50 p-3 rounded-md text-xs overflow-x-auto">
                  {JSON.stringify(response.headers, null, 2)}
                </pre>
              </div>

              {/* Response Body */}
              <div>
                <h4 className="font-medium text-gray-800 mb-2">Response Body</h4>
                <pre className="bg-gray-50 p-3 rounded-md text-xs overflow-x-auto max-h-64 overflow-y-auto">
                  {typeof response.data === 'string' 
                    ? response.data 
                    : JSON.stringify(response.data, null, 2)
                  }
                </pre>
              </div>
            </div>
          ) : (
            <div className="text-center text-gray-500 py-8">
              No response yet. Send a request to see the response here.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default APITester;