// =============================================================================
// FEEDBACK FORM COMPONENT
// File: frontend/customer-chat/src/components/FeedbackForm.jsx
// =============================================================================

import React, { useState } from 'react';
import './FeedbackForm.css';

const FeedbackForm = ({ requestId, onFeedbackSubmit, onClose }) => {
  const [feedbackType, setFeedbackType] = useState('thumbs');
  const [rating, setRating] = useState(0);
  const [thumbs, setThumbs] = useState(null);
  const [comment, setComment] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [submitted, setSubmitted] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (submitting) return;
    
    setSubmitting(true);
    
    try {
      const feedbackData = {
        request_id: requestId,
        feedback_type: feedbackType,
        rating: feedbackType === 'rating' ? rating : null,
        thumbs: feedbackType === 'thumbs' ? thumbs : null,
        comment: comment.trim() || null,
        timestamp: new Date().toISOString()
      };
      
      await onFeedbackSubmit(feedbackData);
      setSubmitted(true);
      
      // Auto-close after 2 seconds
      setTimeout(() => {
        onClose();
      }, 2000);
      
    } catch (error) {
      console.error('Error submitting feedback:', error);
      // Handle error (show message to user)
    } finally {
      setSubmitting(false);
    }
  };

  const handleThumbsClick = (value) => {
    setThumbs(value);
    if (feedbackType === 'thumbs' && !comment.trim()) {
      // Quick submit for thumbs without comment
      setTimeout(() => {
        document.getElementById('feedback-form').requestSubmit();
      }, 100);
    }
  };

  const handleRatingClick = (value) => {
    setRating(value);
  };

  if (submitted) {
    return (
      <div className="feedback-form success">
        <div className="success-message">
          <div className="success-icon">✓</div>
          <h3>Thank you for your feedback!</h3>
          <p>Your feedback helps us improve.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="feedback-form">
      <div className="feedback-header">
        <h3>How was this response?</h3>
        <button className="close-button" onClick={onClose}>×</button>
      </div>
      
      <form id="feedback-form" onSubmit={handleSubmit}>
        <div className="feedback-type-selector">
          <label>
            <input
              type="radio"
              value="thumbs"
              checked={feedbackType === 'thumbs'}
              onChange={(e) => setFeedbackType(e.target.value)}
            />
            Quick Feedback
          </label>
          <label>
            <input
              type="radio"
              value="rating"
              checked={feedbackType === 'rating'}
              onChange={(e) => setFeedbackType(e.target.value)}
            />
            Detailed Rating
          </label>
        </div>
        
        {feedbackType === 'thumbs' && (
          <div className="thumbs-feedback">
            <button
              type="button"
              className={`thumb-button ${thumbs === 'up' ? 'active' : ''}`}
              onClick={() => handleThumbsClick('up')}
            >
              👍 Helpful
            </button>
            <button
              type="button"
              className={`thumb-button ${thumbs === 'down' ? 'active' : ''}`}
              onClick={() => handleThumbsClick('down')}
            >
              👎 Not helpful
            </button>
          </div>
        )}
        
        {feedbackType === 'rating' && (
          <div className="rating-feedback">
            <div className="rating-stars">
              {[1, 2, 3, 4, 5].map((star) => (
                <button
                  key={star}
                  type="button"
                  className={`star-button ${rating >= star ? 'active' : ''}`}
                  onClick={() => handleRatingClick(star)}
                >
                  ★
                </button>
              ))}
            </div>
            <div className="rating-labels">
              <span>Poor</span>
              <span>Excellent</span>
            </div>
          </div>
        )}
        
        <div className="comment-section">
          <label htmlFor="comment">Additional comments (optional):</label>
          <textarea
            id="comment"
            value={comment}
            onChange={(e) => setComment(e.target.value)}
            placeholder="Tell us more about your experience..."
            rows={3}
          />
        </div>
        
        <div className="form-actions">
          <button
            type="button"
            className="cancel-button"
            onClick={onClose}
          >
            Cancel
          </button>
          <button
            type="submit"
            className="submit-button"
            disabled={submitting || (feedbackType === 'thumbs' && !thumbs) || (feedbackType === 'rating' && !rating)}
          >
            {submitting ? 'Submitting...' : 'Submit Feedback'}
          </button>
        </div>
      </form>
    </div>
  );
};

export default FeedbackForm;
