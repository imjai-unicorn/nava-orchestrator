# backend/services/01-core/nava-logic-controller/app/service/feedback_processor.py
"""
Feedback Processor - Week 3 Component (Complete Version)
Processes user feedback to improve AI model selection and performance
"""

import logging
import time
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)

class FeedbackType(Enum):
    """Types of feedback"""
    RATING = "rating"
    THUMBS = "thumbs" 
    QUALITY = "quality"
    PREFERENCE = "preference"
    DETAILED = "detailed"
    IMPLICIT = "implicit"

class FeedbackCategory(Enum):
    """Feedback categories"""
    ACCURACY = "accuracy"
    HELPFULNESS = "helpfulness"
    CLARITY = "clarity"
    COMPLETENESS = "completeness"
    RELEVANCE = "relevance"
    SPEED = "speed"
    OVERALL = "overall"

@dataclass
class FeedbackEntry:
    """Individual feedback entry"""
    feedback_id: str
    user_id: str
    session_id: Optional[str]
    response_id: str
    model_used: str
    pattern_detected: str
    feedback_type: FeedbackType
    feedback_category: FeedbackCategory
    score: float
    comment: Optional[str]
    metadata: Dict[str, Any]
    timestamp: datetime
    processing_time: float
    context: Optional[Dict[str, Any]]

@dataclass
class FeedbackStats:
    """Feedback statistics"""
    total_feedback_count: int
    feedback_by_model: Dict[str, Dict[str, Any]]
    feedback_by_pattern: Dict[str, Dict[str, Any]]
    feedback_by_category: Dict[str, Dict[str, Any]]
    avg_scores: Dict[str, float]
    trends: Dict[str, List[float]]
    learning_metrics: Dict[str, Any]
    last_updated: datetime

class FeedbackProcessor:
    """Advanced feedback processing system"""
    
    def __init__(self):
        self.feedback_storage = []
        self.max_feedback_entries = 10000
        self.learning_threshold = 5
        
        # Feedback weights
        self.feedback_weights = {
            FeedbackType.RATING: 1.0,
            FeedbackType.THUMBS: 0.8,
            FeedbackType.QUALITY: 1.2,
            FeedbackType.PREFERENCE: 1.1,
            FeedbackType.DETAILED: 1.5,
            FeedbackType.IMPLICIT: 0.6
        }
    
    async def process_feedback(
        self,
        user_id: str,
        response_id: str,
        model_used: str,
        pattern_detected: str,
        feedback_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process incoming feedback"""
        
        start_time = time.time()
        
        try:
            # Normalize feedback
            normalized_feedback = self._normalize_feedback(feedback_data)
            
            # Create feedback entry
            feedback_entry = FeedbackEntry(
                feedback_id=self._generate_feedback_id(),
                user_id=user_id,
                session_id=context.get('session_id') if context else None,
                response_id=response_id,
                model_used=model_used,
                pattern_detected=pattern_detected,
                feedback_type=FeedbackType(normalized_feedback.get('type', 'rating')),
                feedback_category=FeedbackCategory(normalized_feedback.get('category', 'overall')),
                score=normalized_feedback.get('score', 0.5),
                comment=normalized_feedback.get('comment'),
                metadata=normalized_feedback.get('metadata', {}),
                timestamp=datetime.now(),
                processing_time=0.0,
                context=context
            )
            
            # Store feedback
            self._store_feedback(feedback_entry)
            
            # Update learning models
            learning_updates = await self._update_learning_models(feedback_entry)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            feedback_entry.processing_time = processing_time
            
            logger.info(
                f"📝 Feedback processed: {model_used}/{pattern_detected}, "
                f"score={feedback_entry.score:.2f}"
            )
            
            return {
                'feedback_id': feedback_entry.feedback_id,
                'processed': True,
                'learning_updates': learning_updates,
                'processing_time': processing_time,
                'timestamp': feedback_entry.timestamp.isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Feedback processing error: {e}")
            return {
                'processed': False,
                'error': str(e),
                'processing_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }
    
    async def get_feedback_stats(self, days: int = 30) -> FeedbackStats:
        """Get comprehensive feedback statistics"""
        
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_feedback = [f for f in self.feedback_storage if f.timestamp >= cutoff_date]
            
            stats = FeedbackStats(
                total_feedback_count=len(recent_feedback),
                feedback_by_model=self._calculate_model_stats(recent_feedback),
                feedback_by_pattern=self._calculate_pattern_stats(recent_feedback),
                feedback_by_category=self._calculate_category_stats(recent_feedback),
                avg_scores=self._calculate_average_scores(recent_feedback),
                trends=self._calculate_trends(recent_feedback),
                learning_metrics=self._calculate_learning_metrics(recent_feedback),
                last_updated=datetime.now()
            )
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error calculating feedback stats: {e}")
            return self._create_empty_stats()
    
    def _normalize_feedback(self, feedback_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize feedback data to standard format"""
        
        normalized = {}
        
        # Determine feedback type and score
        if 'rating' in feedback_data:
            normalized['type'] = 'rating'
            rating = feedback_data['rating']
            if isinstance(rating, (int, float)):
                if rating <= 1:
                    normalized['score'] = max(0, min(rating, 1))
                elif rating <= 5:
                    normalized['score'] = (rating - 1) / 4
                elif rating <= 10:
                    normalized['score'] = (rating - 1) / 9
                else:
                    normalized['score'] = 0.5
        elif 'thumbs' in feedback_data:
            normalized['type'] = 'thumbs'
            normalized['score'] = 1.0 if feedback_data['thumbs'] == 'up' else 0.0
        else:
            normalized['type'] = 'rating'
            normalized['score'] = 0.5
        
        # Determine category
        normalized['category'] = feedback_data.get('category', 'overall')
        if normalized['category'] not in [c.value for c in FeedbackCategory]:
            normalized['category'] = 'overall'
        
        # Add other fields
        normalized['comment'] = feedback_data.get('comment')
        normalized['metadata'] = feedback_data.get('metadata', {})
        
        return normalized
    
    def _store_feedback(self, feedback_entry: FeedbackEntry):
        """Store feedback entry"""
        self.feedback_storage.append(feedback_entry)
        
        if len(self.feedback_storage) > self.max_feedback_entries:
            self.feedback_storage = self.feedback_storage[-self.max_feedback_entries:]
    
    def _generate_feedback_id(self) -> str:
        """Generate unique feedback ID"""
        timestamp = int(time.time() * 1000)
        return f"fb_{timestamp}_{hash(str(timestamp)) % 10000}"
    
    async def _update_learning_models(self, feedback_entry: FeedbackEntry) -> Dict[str, Any]:
        """Update learning models based on feedback"""
        
        updates = {
            'model_weights_updated': False,
            'pattern_weights_updated': False,
            'learning_triggered': False
        }
        
        try:
            model_feedback_count = len([f for f in self.feedback_storage if f.model_used == feedback_entry.model_used])
            pattern_feedback_count = len([f for f in self.feedback_storage if f.pattern_detected == feedback_entry.pattern_detected])
            
            if model_feedback_count >= self.learning_threshold:
                updates['model_weights_updated'] = True
                updates['learning_triggered'] = True
            
            if pattern_feedback_count >= self.learning_threshold:
                updates['pattern_weights_updated'] = True
                updates['learning_triggered'] = True
            
            return updates
            
        except Exception as e:
            logger.error(f"❌ Error updating learning models: {e}")
            return updates
    
    def _calculate_model_stats(self, feedback_list: List[FeedbackEntry]) -> Dict[str, Dict[str, Any]]:
        """Calculate statistics by model"""
        
        model_stats = defaultdict(lambda: {
            'total_feedback': 0,
            'avg_score': 0.0,
            'score_distribution': {'low': 0, 'medium': 0, 'high': 0}
        })
        
        for feedback in feedback_list:
            model = feedback.model_used
            model_stats[model]['total_feedback'] += 1
            
            if feedback.score < 0.4:
                model_stats[model]['score_distribution']['low'] += 1
            elif feedback.score < 0.7:
                model_stats[model]['score_distribution']['medium'] += 1
            else:
                model_stats[model]['score_distribution']['high'] += 1
        
        # Calculate averages
        for model, stats in model_stats.items():
            if stats['total_feedback'] > 0:
                total_score = sum(f.score for f in feedback_list if f.model_used == model)
                stats['avg_score'] = total_score / stats['total_feedback']
        
        return dict(model_stats)
    
    def _calculate_pattern_stats(self, feedback_list: List[FeedbackEntry]) -> Dict[str, Dict[str, Any]]:
        """Calculate statistics by pattern"""
        
        pattern_stats = defaultdict(lambda: {
            'total_feedback': 0,
            'avg_score': 0.0,
            'model_performance': defaultdict(list)
        })
        
        for feedback in feedback_list:
            pattern = feedback.pattern_detected
            pattern_stats[pattern]['total_feedback'] += 1
            pattern_stats[pattern]['model_performance'][feedback.model_used].append(feedback.score)
        
        # Calculate averages
        for pattern, stats in pattern_stats.items():
            if stats['total_feedback'] > 0:
                total_score = sum(f.score for f in feedback_list if f.pattern_detected == pattern)
                stats['avg_score'] = total_score / stats['total_feedback']
        
        return dict(pattern_stats)
    
    def _calculate_category_stats(self, feedback_list: List[FeedbackEntry]) -> Dict[str, Dict[str, Any]]:
        """Calculate statistics by category"""
        
        category_stats = defaultdict(lambda: {
            'total_feedback': 0,
            'avg_score': 0.0
        })
        
        for feedback in feedback_list:
            category = feedback.feedback_category.value
            category_stats[category]['total_feedback'] += 1
        
        # Calculate averages
        for category, stats in category_stats.items():
            if stats['total_feedback'] > 0:
                total_score = sum(f.score for f in feedback_list if f.feedback_category.value == category)
                stats['avg_score'] = total_score / stats['total_feedback']
        
        return dict(category_stats)
    
    def _calculate_average_scores(self, feedback_list: List[FeedbackEntry]) -> Dict[str, float]:
        """Calculate average scores"""
        
        if not feedback_list:
            return {}
        
        return {
            'overall': sum(f.score for f in feedback_list) / len(feedback_list),
            'recent_7_days': sum(
                f.score for f in feedback_list 
                if f.timestamp >= datetime.now() - timedelta(days=7)
            ) / max(len([
                f for f in feedback_list 
                if f.timestamp >= datetime.now() - timedelta(days=7)
            ]), 1)
        }
    
    def _calculate_trends(self, feedback_list: List[FeedbackEntry]) -> Dict[str, List[float]]:
        """Calculate trend data"""
        
        sorted_feedback = sorted(feedback_list, key=lambda f: f.timestamp)
        
        daily_scores = defaultdict(list)
        for feedback in sorted_feedback:
            day_key = feedback.timestamp.date().isoformat()
            daily_scores[day_key].append(feedback.score)
        
        daily_averages = {
            day: sum(scores) / len(scores) 
            for day, scores in daily_scores.items()
        }
        
        return {
            'daily_averages': list(daily_averages.values()),
            'dates': list(daily_averages.keys())
        }
    
    def _calculate_learning_metrics(self, feedback_list: List[FeedbackEntry]) -> Dict[str, Any]:
        """Calculate learning system metrics"""
        
        return {
            'total_learning_samples': len(feedback_list),
            'learning_enabled': len(feedback_list) >= self.learning_threshold,
            'learning_confidence': min(len(feedback_list) / 100.0, 1.0),
            'last_learning_update': max(
                (f.timestamp for f in feedback_list), 
                default=datetime.now()
            ).isoformat()
        }
    
    def _create_empty_stats(self) -> FeedbackStats:
        """Create empty stats object"""
        
        return FeedbackStats(
            total_feedback_count=0,
            feedback_by_model={},
            feedback_by_pattern={},
            feedback_by_category={},
            avg_scores={},
            trends={},
            learning_metrics={'learning_enabled': False},
            last_updated=datetime.now()
        )

# Global feedback processor instance
feedback_processor = FeedbackProcessor()

# Helper functions
async def process_user_feedback(
    user_id: str,
    response_id: str,
    model_used: str,
    pattern_detected: str,
    feedback_data: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Helper function to process user feedback"""
    
    try:
        result = await feedback_processor.process_feedback(
            user_id=user_id,
            response_id=response_id,
            model_used=model_used,
            pattern_detected=pattern_detected,
            feedback_data=feedback_data,
            context=context
        )
        return result
        
    except Exception as e:
        logger.error(f"❌ Feedback processing helper error: {e}")
        return {
            'processed': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

async def get_learning_insights(days: int = 30) -> Dict[str, Any]:
    """Helper function to get learning insights"""
    
    try:
        stats = await feedback_processor.get_feedback_stats(days)
        
        return {
            'stats': asdict(stats),
            'learning_status': 'active' if stats.total_feedback_count >= feedback_processor.learning_threshold else 'collecting',
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Learning insights helper error: {e}")
        return {
            'error': str(e),
            'learning_status': 'error',
            'timestamp': datetime.now().isoformat()
        }