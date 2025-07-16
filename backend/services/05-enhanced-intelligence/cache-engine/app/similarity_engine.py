# backend/services/05-enhanced-intelligence/cache-engine/app/similarity_engine.py
"""
Similarity Engine
Semantic similarity matching for intelligent caching
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Tuple
import logging
import time
import re
import math
from datetime import datetime
from collections import Counter

logger = logging.getLogger(__name__)

# Create router
similarity_router = APIRouter()

# Models
class SimilarityRequest(BaseModel):
    query: str = Field(..., description="Query to find similar cached responses")
    threshold: float = Field(default=0.8, description="Similarity threshold (0.0-1.0)")
    max_results: int = Field(default=5, description="Maximum number of similar results")
    include_scores: bool = Field(default=True, description="Include similarity scores")
    similarity_method: str = Field(default="hybrid", description="cosine, jaccard, levenshtein, hybrid")

class SimilarityResponse(BaseModel):
    query: str
    similar_entries: List[Dict[str, Any]]
    total_found: int
    search_method: str
    processing_time_seconds: float
    timestamp: str

class TextSimilarityRequest(BaseModel):
    text1: str = Field(..., description="First text")
    text2: str = Field(..., description="Second text")
    method: str = Field(default="hybrid", description="Similarity calculation method")

class TextSimilarityResponse(BaseModel):
    text1_preview: str
    text2_preview: str
    similarity_score: float
    method_used: str
    detailed_scores: Dict[str, float]
    is_similar: bool
    processing_time_seconds: float

class SimilarityEngine:
    """Advanced Similarity Engine for Semantic Matching"""
    
    def __init__(self):
        # Similarity configurations
        self.similarity_config = {
            "cosine": {
                "weight": 0.4,
                "description": "Cosine similarity based on term frequency",
                "threshold": 0.7
            },
            "jaccard": {
                "weight": 0.3,
                "description": "Jaccard similarity based on word overlap",
                "threshold": 0.6
            },
            "levenshtein": {
                "weight": 0.2,
                "description": "Levenshtein distance for string similarity",
                "threshold": 0.8
            },
            "semantic": {
                "weight": 0.1,
                "description": "Basic semantic similarity",
                "threshold": 0.7
            }
        }
        
        # Stop words for text processing
        self.stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "up", "about", "into", "through", "during",
            "before", "after", "above", "below", "under", "between", "among",
            "is", "am", "are", "was", "were", "be", "been", "being", "have",
            "has", "had", "do", "does", "did", "will", "would", "could", "should",
            "may", "might", "must", "can", "this", "that", "these", "those"
        }
        
        # Cached similarity calculations
        self.similarity_cache = {}
        self.cache_max_size = 1000
        
        # Statistics
        self.stats = {
            "similarity_calculations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "average_processing_time": 0.0
        }
    
    async def find_similar(self, request: SimilarityRequest, 
                          cached_entries: Dict[str, Any]) -> Dict[str, Any]:
        """Find similar cached entries to the query"""
        
        start_time = time.time()
        
        try:
            # Preprocess query
            processed_query = self._preprocess_text(request.query)
            
            # Calculate similarities
            similarities = []
            
            for cache_key, cache_data in cached_entries.items():
                # Extract text from cache data
                cache_text = self._extract_cache_text(cache_data)
                if not cache_text:
                    continue
                
                # Calculate similarity
                similarity_score = await self._calculate_similarity(
                    processed_query,
                    self._preprocess_text(cache_text),
                    request.similarity_method
                )
                
                # Check threshold
                if similarity_score >= request.threshold:
                    similarity_entry = {
                        "cache_key": cache_key,
                        "cache_text_preview": cache_text[:200] + "..." if len(cache_text) > 200 else cache_text,
                        "similarity_score": round(similarity_score, 4),
                        "cache_metadata": self._extract_cache_metadata(cache_data)
                    }
                    
                    if request.include_scores:
                        similarity_entry["detailed_scores"] = await self._get_detailed_scores(
                            processed_query, 
                            self._preprocess_text(cache_text)
                        )
                    
                    similarities.append(similarity_entry)
            
            # Sort by similarity score (highest first)
            similarities.sort(key=lambda x: x["similarity_score"], reverse=True)
            
            # Limit results
            limited_similarities = similarities[:request.max_results]
            
            processing_time = time.time() - start_time
            self._update_stats(processing_time)
            
            return {
                "similar_entries": limited_similarities,
                "total_found": len(similarities),
                "search_method": request.similarity_method,
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"❌ Find similar error: {e}")
            return {
                "similar_entries": [],
                "total_found": 0,
                "search_method": request.similarity_method,
                "processing_time": time.time() - start_time,
                "error": str(e)
            }
    
    async def calculate_text_similarity(self, request: TextSimilarityRequest) -> Dict[str, Any]:
        """Calculate similarity between two texts"""
        
        start_time = time.time()
        
        try:
            # Preprocess texts
            text1_processed = self._preprocess_text(request.text1)
            text2_processed = self._preprocess_text(request.text2)
            
            # Calculate similarity
            similarity_score = await self._calculate_similarity(
                text1_processed,
                text2_processed,
                request.method
            )
            
            # Get detailed scores
            detailed_scores = await self._get_detailed_scores(
                text1_processed,
                text2_processed
            )
            
            # Determine if similar
            threshold = self.similarity_config.get(request.method, {}).get("threshold", 0.7)
            is_similar = similarity_score >= threshold
            
            processing_time = time.time() - start_time
            
            return {
                "text1_preview": request.text1[:100] + "..." if len(request.text1) > 100 else request.text1,
                "text2_preview": request.text2[:100] + "..." if len(request.text2) > 100 else request.text2,
                "similarity_score": round(similarity_score, 4),
                "method_used": request.method,
                "detailed_scores": detailed_scores,
                "is_similar": is_similar,
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"❌ Text similarity error: {e}")
            return {
                "similarity_score": 0.0,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for similarity calculation"""
        
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and normalize whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Remove stop words
        words = text.split()
        filtered_words = [word for word in words if word not in self.stop_words and len(word) > 2]
        
        return ' '.join(filtered_words)
    
    async def _calculate_similarity(self, text1: str, text2: str, method: str) -> float:
        """Calculate similarity using specified method"""
        
        # Check cache first
        cache_key = self._get_similarity_cache_key(text1, text2, method)
        if cache_key in self.similarity_cache:
            self.stats["cache_hits"] += 1
            return self.similarity_cache[cache_key]
        
        self.stats["cache_misses"] += 1
        self.stats["similarity_calculations"] += 1
        
        try:
            if method == "cosine":
                similarity = self._cosine_similarity(text1, text2)
            elif method == "jaccard":
                similarity = self._jaccard_similarity(text1, text2)
            elif method == "levenshtein":
                similarity = self._levenshtein_similarity(text1, text2)
            elif method == "semantic":
                similarity = self._semantic_similarity(text1, text2)
            elif method == "hybrid":
                similarity = await self._hybrid_similarity(text1, text2)
            else:
                similarity = await self._hybrid_similarity(text1, text2)
            
            # Cache result
            self._cache_similarity(cache_key, similarity)
            
            return similarity
            
        except Exception as e:
            logger.error(f"❌ Similarity calculation error: {e}")
            return 0.0
    
    def _cosine_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts"""
        
        if not text1 or not text2:
            return 0.0
        
        # Get word frequency vectors
        words1 = text1.split()
        words2 = text2.split()
        
        # Create vocabulary
        vocabulary = set(words1 + words2)
        
        if not vocabulary:
            return 0.0
        
        # Create frequency vectors
        vector1 = [words1.count(word) for word in vocabulary]
        vector2 = [words2.count(word) for word in vocabulary]
        
        # Calculate cosine similarity
        dot_product = sum(a * b for a, b in zip(vector1, vector2))
        magnitude1 = math.sqrt(sum(a * a for a in vector1))
        magnitude2 = math.sqrt(sum(b * b for b in vector2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity between two texts"""
        
        if not text1 or not text2:
            return 0.0
        
        set1 = set(text1.split())
        set2 = set(text2.split())
        
        if not set1 and not set2:
            return 1.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def _levenshtein_similarity(self, text1: str, text2: str) -> float:
        """Calculate normalized Levenshtein similarity"""
        
        if not text1 or not text2:
            return 0.0 if text1 != text2 else 1.0
        
        # Calculate Levenshtein distance
        distance = self._levenshtein_distance(text1, text2)
        
        # Normalize to similarity (0-1)
        max_len = max(len(text1), len(text2))
        similarity = 1 - (distance / max_len) if max_len > 0 else 1.0
        
        return max(0.0, similarity)
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings"""
        
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def _semantic_similarity(self, text1: str, text2: str) -> float:
        """Basic semantic similarity using word overlap and context"""
        
        if not text1 or not text2:
            return 0.0
        
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        # Direct word overlap
        overlap = len(words1.intersection(words2))
        total_unique = len(words1.union(words2))
        
        base_similarity = overlap / total_unique if total_unique > 0 else 0.0
        
        # Boost for similar length
        len_ratio = min(len(text1), len(text2)) / max(len(text1), len(text2)) if max(len(text1), len(text2)) > 0 else 1.0
        length_bonus = len_ratio * 0.1
        
        # Boost for common patterns
        pattern_bonus = 0.0
        common_patterns = ["how to", "what is", "can you", "please explain"]
        
        for pattern in common_patterns:
            if pattern in text1.lower() and pattern in text2.lower():
                pattern_bonus += 0.05
        
        return min(1.0, base_similarity + length_bonus + pattern_bonus)
    
    async def _hybrid_similarity(self, text1: str, text2: str) -> float:
        """Hybrid similarity combining multiple methods"""
        
        # Calculate individual similarities
        cosine_sim = self._cosine_similarity(text1, text2)
        jaccard_sim = self._jaccard_similarity(text1, text2)
        levenshtein_sim = self._levenshtein_similarity(text1, text2)
        semantic_sim = self._semantic_similarity(text1, text2)
        
        # Weighted combination
        hybrid_score = (
            cosine_sim * self.similarity_config["cosine"]["weight"] +
            jaccard_sim * self.similarity_config["jaccard"]["weight"] +
            levenshtein_sim * self.similarity_config["levenshtein"]["weight"] +
            semantic_sim * self.similarity_config["semantic"]["weight"]
        )
        
        return min(1.0, max(0.0, hybrid_score))
    
    async def _get_detailed_scores(self, text1: str, text2: str) -> Dict[str, float]:
        """Get detailed scores for all similarity methods"""
        
        return {
            "cosine": round(self._cosine_similarity(text1, text2), 4),
            "jaccard": round(self._jaccard_similarity(text1, text2), 4),
            "levenshtein": round(self._levenshtein_similarity(text1, text2), 4),
            "semantic": round(self._semantic_similarity(text1, text2), 4),
            "hybrid": round(await self._hybrid_similarity(text1, text2), 4)
        }
    
    def _extract_cache_text(self, cache_data: Dict[str, Any]) -> str:
        """Extract text content from cache data"""
        
        if isinstance(cache_data, dict):
            # Try common text fields
            for field in ["text", "content", "response", "value", "message", "original_value"]:
                if field in cache_data:
                    value = cache_data[field]
                    if isinstance(value, str):
                        return value
                    elif isinstance(value, dict) and "text" in value:
                        return value["text"]
            
            # Convert dict to string as fallback
            return str(cache_data)
        
        elif isinstance(cache_data, str):
            return cache_data
        
        else:
            return str(cache_data)
    
    def _extract_cache_metadata(self, cache_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from cache data"""
        
        metadata = {}
        
        if isinstance(cache_data, dict):
            # Common metadata fields
            for field in ["created_at", "last_accessed", "access_count", "size_bytes", "tags", "priority"]:
                if field in cache_data:
                    metadata[field] = cache_data[field]
        
        return metadata
    
    def _get_similarity_cache_key(self, text1: str, text2: str, method: str) -> str:
        """Generate cache key for similarity calculation"""
        
        # Sort texts to ensure consistent cache key
        sorted_texts = sorted([text1, text2])
        combined = f"{sorted_texts[0]}|{sorted_texts[1]}|{method}"
        
        # Hash for shorter key
        import hashlib
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _cache_similarity(self, cache_key: str, similarity: float):
        """Cache similarity calculation result"""
        
        # Implement LRU eviction if cache is full
        if len(self.similarity_cache) >= self.cache_max_size:
            # Remove oldest entry (simple implementation)
            oldest_key = next(iter(self.similarity_cache))
            del self.similarity_cache[oldest_key]
        
        self.similarity_cache[cache_key] = similarity
    
    def _update_stats(self, processing_time: float):
        """Update processing statistics"""
        
        current_avg = self.stats["average_processing_time"]
        calculations = self.stats["similarity_calculations"]
        
        # Update running average
        self.stats["average_processing_time"] = (
            (current_avg * (calculations - 1) + processing_time) / calculations
            if calculations > 0 else processing_time
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get similarity engine statistics"""
        
        cache_hit_rate = (
            self.stats["cache_hits"] / max(self.stats["cache_hits"] + self.stats["cache_misses"], 1)
        )
        
        return {
            "similarity_calculations": self.stats["similarity_calculations"],
            "cache_hits": self.stats["cache_hits"],
            "cache_misses": self.stats["cache_misses"],
            "cache_hit_rate": round(cache_hit_rate, 4),
            "average_processing_time": round(self.stats["average_processing_time"], 6),
            "cached_similarities": len(self.similarity_cache),
            "supported_methods": list(self.similarity_config.keys()),
            "default_method": "hybrid"
        }

# Initialize similarity engine
similarity_engine = SimilarityEngine()

@similarity_router.post("/find-similar", response_model=SimilarityResponse)
async def find_similar_entries(request: SimilarityRequest):
    """Find similar cached entries"""
    
    start_time = time.time()
    
    try:
        # Get cached entries (mock implementation)
        # In real implementation, this would query the actual cache
        cached_entries = {}  # This should be populated from actual cache
        
        # Find similar entries
        result = await similarity_engine.find_similar(request, cached_entries)
        
        processing_time = time.time() - start_time
        
        response = SimilarityResponse(
            query=request.query,
            similar_entries=result.get("similar_entries", []),
            total_found=result.get("total_found", 0),
            search_method=result.get("search_method", request.similarity_method),
            processing_time_seconds=processing_time,
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(
            f"🔍 Similarity search: '{request.query[:50]}...' → "
            f"{result.get('total_found', 0)} matches ({result.get('search_method', 'unknown')})"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Find similar entries error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@similarity_router.post("/compare-texts", response_model=TextSimilarityResponse)
async def compare_texts(request: TextSimilarityRequest):
    """Compare similarity between two texts"""
    
    try:
        result = await similarity_engine.calculate_text_similarity(request)
        
        response = TextSimilarityResponse(**result)
        
        logger.info(
            f"📊 Text similarity: {result.get('similarity_score', 0):.4f} "
            f"({result.get('method_used', 'unknown')})"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Compare texts error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@similarity_router.get("/methods")
async def get_similarity_methods():
    """Get available similarity methods"""
    
    return {
        "available_methods": list(similarity_engine.similarity_config.keys()),
        "method_details": similarity_engine.similarity_config,
        "default_method": "hybrid",
        "recommended_thresholds": {
            "high_similarity": 0.8,
            "medium_similarity": 0.6,
            "low_similarity": 0.4
        },
        "timestamp": datetime.now().isoformat()
    }

@similarity_router.get("/stats")
async def get_similarity_stats():
    """Get similarity engine statistics"""
    
    try:
        stats = similarity_engine.get_stats()
        
        return {
            **stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Similarity stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@similarity_router.post("/benchmark")
async def benchmark_similarity_methods(text1: str, text2: str):
    """Benchmark all similarity methods"""
    
    try:
        benchmark_results = {}
        
        for method in similarity_engine.similarity_config.keys():
            start_time = time.time()
            
            similarity_score = await similarity_engine._calculate_similarity(
                similarity_engine._preprocess_text(text1),
                similarity_engine._preprocess_text(text2),
                method
            )
            
            processing_time = time.time() - start_time
            
            benchmark_results[method] = {
                "similarity_score": round(similarity_score, 4),
                "processing_time_ms": round(processing_time * 1000, 2),
                "threshold": similarity_engine.similarity_config[method]["threshold"]
            }
        
        # Find best method
        best_method = max(benchmark_results.items(), key=lambda x: x[1]["similarity_score"])
        
        return {
            "text1_preview": text1[:100] + "..." if len(text1) > 100 else text1,
            "text2_preview": text2[:100] + "..." if len(text2) > 100 else text2,
            "method_results": benchmark_results,
            "best_method": {
                "method": best_method[0],
                "score": best_method[1]["similarity_score"]
            },
            "total_processing_time_ms": sum(r["processing_time_ms"] for r in benchmark_results.values()),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Benchmark similarity methods error: {e}")
        raise HTTPException(status_code=500, detail=str(e))