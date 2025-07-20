/ =============================================================================
// LOADING SPINNER COMPONENT
// File: frontend/customer-chat/src/components/LoadingSpinner.jsx
// =============================================================================

import React from 'react';
import './LoadingSpinner.css';

const LoadingSpinner = ({ 
  size = 'medium', 
  color = 'blue', 
  text = 'Loading...', 
  showText = true,
  inline = false 
}) => {
  const sizeClasses = {
    small: 'w-4 h-4',
    medium: 'w-8 h-8',
    large: 'w-12 h-12'
  };

  const colorClasses = {
    blue: 'border-blue-500',
    green: 'border-green-500',
    red: 'border-red-500',
    gray: 'border-gray-500'
  };

  return (
    <div className={`loading-spinner ${inline ? 'inline' : 'block'}`}>
      <div className={`spinner ${sizeClasses[size]} ${colorClasses[color]}`}>
        <div className="spinner-circle"></div>
      </div>
      {showText && <span className="loading-text">{text}</span>}
    </div>
  );
};

export default LoadingSpinner;