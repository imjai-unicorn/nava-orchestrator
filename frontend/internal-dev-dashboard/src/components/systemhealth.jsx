// frontend/internal-dev-dashboard/src/components/SystemHealth.jsx
import React, { useState, useEffect } from 'react';

const SystemHealth = () => {
  const [healthData, setHealthData] = useState({
    services: [],
    overall: 'loading',
    lastUpdate: null
  });

  useEffect(() => {
    const fetchHealthData = async () => {
      try {
        const response = await fetch('http://localhost:8005/api/health');
        const data = await response.json();
        setHealthData({
          services: [
            { name: 'NAVA Controller', status: data.nava_controller || 'healthy', port: 8005 },
            { name: 'GPT Client', status: data.gpt_client || 'healthy', port: 8002 },
            { name: 'Claude Client', status: data.claude_client || 'healthy', port: 8003 },
            { name: 'Gemini Client', status: data.gemini_client || 'healthy', port: 8004 },
            { name: 'Database', status: data.database || 'healthy', port: 'DB' },
            { name: 'Cache', status: data.cache || 'healthy', port: 'Redis' }
          ],
          overall: data.overall || 'healthy',
          lastUpdate: new Date()
        });
      } catch (error) {
        console.error('Failed to fetch health data:', error);
        setHealthData(prev => ({ ...prev, overall: 'error' }));
      }
    };

    fetchHealthData();
    const interval = setInterval(fetchHealthData, 30000); // Update every 30 seconds

    return () => clearInterval(interval);
  }, []);

  const getStatusColor = (status) => {
    switch (status) {
      case 'healthy': return 'bg-green-500';
      case 'warning': return 'bg-yellow-500';
      case 'error': return 'bg-red-500';
      default: return 'bg-gray-500';
    }
  };

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-bold text-gray-800">System Health</h2>
        <div className={`px-3 py-1 rounded-full text-white text-sm ${getStatusColor(healthData.overall)}`}>
          {healthData.overall.toUpperCase()}
        </div>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {healthData.services.map((service, index) => (
          <div key={index} className="border rounded-lg p-4">
            <div className="flex justify-between items-center mb-2">
              <h3 className="font-semibold text-gray-700">{service.name}</h3>
              <span className="text-sm text-gray-500">:{service.port}</span>
            </div>
            <div className="flex items-center">
              <div className={`w-3 h-3 rounded-full mr-2 ${getStatusColor(service.status)}`}></div>
              <span className="text-sm capitalize">{service.status}</span>
            </div>
          </div>
        ))}
      </div>
      
      {healthData.lastUpdate && (
        <div className="mt-4 text-xs text-gray-500">
          Last updated: {healthData.lastUpdate.toLocaleTimeString()}
        </div>
      )}
    </div>
  );
};

export default SystemHealth;