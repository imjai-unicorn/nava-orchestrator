// frontend/internal-dev-dashboard/src/components/PerformanceMetrics.jsx
import React, { useState, useEffect } from 'react';

const PerformanceMetrics = () => {
  const [metrics, setMetrics] = useState({
    responseTime: { current: 0, p50: 0, p95: 0, p99: 0 },
    requestsPerMinute: 0,
    errorRate: 0,
    cacheHitRate: 0,
    aiServiceMetrics: {
      gpt: { avgTime: 0, successRate: 0 },
      claude: { avgTime: 0, successRate: 0 },
      gemini: { avgTime: 0, successRate: 0 }
    },
    systemResources: {
      cpu: 0,
      memory: 0,
      activeConnections: 0
    }
  });

  const [timeRange, setTimeRange] = useState('1h');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const response = await fetch(`http://localhost:8005/api/admin/metrics?range=${timeRange}`);
        if (response.ok) {
          const data = await response.json();
          setMetrics(data);
        } else {
          // Fallback with mock data for development
          setMetrics({
            responseTime: { current: 1850, p50: 1200, p95: 2800, p99: 4200 },
            requestsPerMinute: 42,
            errorRate: 2.1,
            cacheHitRate: 67.5,
            aiServiceMetrics: {
              gpt: { avgTime: 2100, successRate: 97.8 },
              claude: { avgTime: 1950, successRate: 98.5 },
              gemini: { avgTime: 2250, successRate: 96.2 }
            },
            systemResources: {
              cpu: 45.2,
              memory: 62.1,
              activeConnections: 28
            }
          });
        }
      } catch (error) {
        console.error('Failed to fetch metrics:', error);
        // Use mock data on error
        setMetrics({
          responseTime: { current: 1850, p50: 1200, p95: 2800, p99: 4200 },
          requestsPerMinute: 42,
          errorRate: 2.1,
          cacheHitRate: 67.5,
          aiServiceMetrics: {
            gpt: { avgTime: 2100, successRate: 97.8 },
            claude: { avgTime: 1950, successRate: 98.5 },
            gemini: { avgTime: 2250, successRate: 96.2 }
          },
          systemResources: {
            cpu: 45.2,
            memory: 62.1,
            activeConnections: 28
          }
        });
      } finally {
        setLoading(false);
      }
    };

    fetchMetrics();
    const interval = setInterval(fetchMetrics, 30000); // Update every 30 seconds

    return () => clearInterval(interval);
  }, [timeRange]);

  const getStatusColor = (value, thresholds) => {
    if (value >= thresholds.danger) return 'text-red-600';
    if (value >= thresholds.warning) return 'text-yellow-600';
    return 'text-green-600';
  };

  const getProgressColor = (value, thresholds) => {
    if (value >= thresholds.danger) return 'bg-red-500';
    if (value >= thresholds.warning) return 'bg-yellow-500';
    return 'bg-green-500';
  };

  const formatTime = (ms) => {
    if (ms < 1000) return `${ms}ms`;
    return `${(ms / 1000).toFixed(1)}s`;
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
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-xl font-bold text-gray-800">Performance Metrics</h2>
        <select
          value={timeRange}
          onChange={(e) => setTimeRange(e.target.value)}
          className="px-3 py-1 border border-gray-300 rounded-md text-sm focus:ring-blue-500 focus:border-blue-500"
        >
          <option value="1h">Last Hour</option>
          <option value="6h">Last 6 Hours</option>
          <option value="24h">Last 24 Hours</option>
          <option value="7d">Last 7 Days</option>
        </select>
      </div>

      {/* Key Metrics Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <div className="bg-blue-50 p-4 rounded-lg">
          <div className="text-sm text-blue-600 font-medium">Avg Response Time</div>
          <div className={`text-2xl font-bold ${getStatusColor(metrics.responseTime.current, { warning: 2000, danger: 3000 })}`}>
            {formatTime(metrics.responseTime.current)}
          </div>
          <div className="text-xs text-gray-500">Target: &lt;2s</div>
        </div>

        <div className="bg-green-50 p-4 rounded-lg">
          <div className="text-sm text-green-600 font-medium">Requests/Min</div>
          <div className="text-2xl font-bold text-green-800">
            {metrics.requestsPerMinute}
          </div>
          <div className="text-xs text-gray-500">Peak capacity: 1000</div>
        </div>

        <div className="bg-red-50 p-4 rounded-lg">
          <div className="text-sm text-red-600 font-medium">Error Rate</div>
          <div className={`text-2xl font-bold ${getStatusColor(metrics.errorRate, { warning: 5, danger: 10 })}`}>
            {metrics.errorRate.toFixed(1)}%
          </div>
          <div className="text-xs text-gray-500">Target: &lt;1%</div>
        </div>

        <div className="bg-purple-50 p-4 rounded-lg">
          <div className="text-sm text-purple-600 font-medium">Cache Hit Rate</div>
          <div className={`text-2xl font-bold ${getStatusColor(100 - metrics.cacheHitRate, { warning: 50, danger: 70 })}`}>
            {metrics.cacheHitRate.toFixed(1)}%
          </div>
          <div className="text-xs text-gray-500">Target: &gt;40%</div>
        </div>
      </div>

      {/* Response Time Percentiles */}
      <div className="mb-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-3">Response Time Distribution</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-gray-50 p-3 rounded">
            <div className="text-sm text-gray-600">P50 (Median)</div>
            <div className="text-xl font-bold">{formatTime(metrics.responseTime.p50)}</div>
          </div>
          <div className="bg-gray-50 p-3 rounded">
            <div className="text-sm text-gray-600">P95</div>
            <div className={`text-xl font-bold ${getStatusColor(metrics.responseTime.p95, { warning: 3000, danger: 5000 })}`}>
              {formatTime(metrics.responseTime.p95)}
            </div>
          </div>
          <div className="bg-gray-50 p-3 rounded">
            <div className="text-sm text-gray-600">P99</div>
            <div className={`text-xl font-bold ${getStatusColor(metrics.responseTime.p99, { warning: 5000, danger: 8000 })}`}>
              {formatTime(metrics.responseTime.p99)}
            </div>
          </div>
        </div>
      </div>

      {/* AI Service Performance */}
      <div className="mb-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-3">AI Service Performance</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {Object.entries(metrics.aiServiceMetrics).map(([service, data]) => (
            <div key={service} className="border rounded-lg p-4">
              <div className="flex justify-between items-center mb-2">
                <h4 className="font-medium text-gray-700 capitalize">{service}</h4>
                <span className={`text-sm font-medium ${getStatusColor(100 - data.successRate, { warning: 5, danger: 10 })}`}>
                  {data.successRate.toFixed(1)}%
                </span>
              </div>
              <div className="space-y-2">
                <div>
                  <div className="flex justify-between text-sm">
                    <span>Avg Response</span>
                    <span>{formatTime(data.avgTime)}</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className={`h-2 rounded-full ${getProgressColor(data.avgTime, { warning: 2000, danger: 3000 })}`}
                      style={{ width: `${Math.min((data.avgTime / 5000) * 100, 100)}%` }}
                    ></div>
                  </div>
                </div>
                <div>
                  <div className="flex justify-between text-sm">
                    <span>Success Rate</span>
                    <span>{data.successRate.toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className={`h-2 rounded-full ${getProgressColor(100 - data.successRate, { warning: 5, danger: 10 })}`}
                      style={{ width: `${data.successRate}%` }}
                    ></div>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* System Resources */}
      <div>
        <h3 className="text-lg font-semibold text-gray-800 mb-3">System Resources</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-gray-50 p-4 rounded-lg">
            <div className="flex justify-between items-center mb-2">
              <span className="text-sm font-medium text-gray-700">CPU Usage</span>
              <span className={`text-sm font-bold ${getStatusColor(metrics.systemResources.cpu, { warning: 70, danger: 85 })}`}>
                {metrics.systemResources.cpu.toFixed(1)}%
              </span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-3">
              <div 
                className={`h-3 rounded-full transition-all duration-300 ${getProgressColor(metrics.systemResources.cpu, { warning: 70, danger: 85 })}`}
                style={{ width: `${metrics.systemResources.cpu}%` }}
              ></div>
            </div>
          </div>

          <div className="bg-gray-50 p-4 rounded-lg">
            <div className="flex justify-between items-center mb-2">
              <span className="text-sm font-medium text-gray-700">Memory Usage</span>
              <span className={`text-sm font-bold ${getStatusColor(metrics.systemResources.memory, { warning: 75, danger: 90 })}`}>
                {metrics.systemResources.memory.toFixed(1)}%
              </span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-3">
              <div 
                className={`h-3 rounded-full transition-all duration-300 ${getProgressColor(metrics.systemResources.memory, { warning: 75, danger: 90 })}`}
                style={{ width: `${metrics.systemResources.memory}%` }}
              ></div>
            </div>
          </div>

          <div className="bg-gray-50 p-4 rounded-lg">
            <div className="flex justify-between items-center mb-2">
              <span className="text-sm font-medium text-gray-700">Active Connections</span>
              <span className="text-sm font-bold text-gray-800">
                {metrics.systemResources.activeConnections}
              </span>
            </div>
            <div className="text-xs text-gray-500">
              Concurrent user sessions
            </div>
          </div>
        </div>
      </div>

      {/* Performance Alerts */}
      <div className="mt-6 p-4 bg-yellow-50 border-l-4 border-yellow-400 rounded">
        <div className="flex items-center">
          <div className="flex-shrink-0">
            <svg className="h-5 w-5 text-yellow-400" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
            </svg>
          </div>
          <div className="ml-3">
            <p className="text-sm text-yellow-700">
              <strong>Performance Notice:</strong> {
                metrics.responseTime.p95 > 3000 ? 'P95 response time exceeds target (3s)' :
                metrics.errorRate > 5 ? 'Error rate is above acceptable threshold (5%)' :
                metrics.cacheHitRate < 40 ? 'Cache hit rate is below target (40%)' :
                'All systems performing within acceptable parameters'
              }
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PerformanceMetrics;