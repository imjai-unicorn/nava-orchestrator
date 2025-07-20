/**
 * Client-side Cache Service
 * File: frontend/customer-chat/src/services/cache.js
 */

class ClientCache {
  constructor(maxSize = 100, ttl = 5 * 60 * 1000) { // 5 minutes default TTL
    this.cache = new Map();
    this.maxSize = maxSize;
    this.ttl = ttl;
    this.stats = {
      hits: 0,
      misses: 0,
      sets: 0,
      evictions: 0
    };
  }

  // Generate cache key
  generateKey(url, params = {}) {
    const sortedParams = Object.keys(params)
      .sort()
      .map(key => `${key}=${params[key]}`)
      .join('&');
    
    return `${url}${sortedParams ? '?' + sortedParams : ''}`;
  }

  // Get item from cache
  get(key) {
    const item = this.cache.get(key);
    
    if (!item) {
      this.stats.misses++;
      return null;
    }

    // Check if expired
    if (Date.now() > item.expires) {
      this.cache.delete(key);
      this.stats.misses++;
      return null;
    }

    // Update access time
    item.lastAccessed = Date.now();
    this.stats.hits++;
    
    return item.data;
  }

  // Set item in cache
  set(key, data, customTtl = null) {
    const ttl = customTtl || this.ttl;
    const expires = Date.now() + ttl;
    
    // If cache is full, remove oldest item
    if (this.cache.size >= this.maxSize && !this.cache.has(key)) {
      this.evictOldest();
    }

    const item = {
      data,
      expires,
      created: Date.now(),
      lastAccessed: Date.now()
    };

    this.cache.set(key, item);
    this.stats.sets++;
  }

  // Remove item from cache
  remove(key) {
    return this.cache.delete(key);
  }

  // Clear all cache
  clear() {
    this.cache.clear();
    this.stats = {
      hits: 0,
      misses: 0,
      sets: 0,
      evictions: 0
    };
  }

  // Evict oldest item
  evictOldest() {
    let oldestKey = null;
    let oldestTime = Infinity;

    for (const [key, item] of this.cache) {
      if (item.lastAccessed < oldestTime) {
        oldestTime = item.lastAccessed;
        oldestKey = key;
      }
    }

    if (oldestKey) {
      this.cache.delete(oldestKey);
      this.stats.evictions++;
    }
  }

  // Get cache statistics
  getStats() {
    const total = this.stats.hits + this.stats.misses;
    const hitRate = total > 0 ? (this.stats.hits / total) * 100 : 0;

    return {
      ...this.stats,
      hitRate: hitRate.toFixed(2),
      size: this.cache.size,
      maxSize: this.maxSize
    };
  }

  // Clean expired items
  cleanup() {
    const now = Date.now();
    const toDelete = [];

    for (const [key, item] of this.cache) {
      if (now > item.expires) {
        toDelete.push(key);
      }
    }

    toDelete.forEach(key => this.cache.delete(key));
    
    return toDelete.length;
  }
}

// Enhanced API client with caching
class CachedAPIClient {
  constructor(baseURL = '/api', cacheConfig = {}) {
    this.baseURL = baseURL;
    this.cache = new ClientCache(
      cacheConfig.maxSize || 100,
      cacheConfig.ttl || 5 * 60 * 1000
    );
    
    // Setup periodic cleanup
    setInterval(() => {
      this.cache.cleanup();
    }, 60000); // Clean every minute
  }

  // GET request with caching
  async get(endpoint, params = {}, options = {}) {
    const url = `${this.baseURL}${endpoint}`;
    const cacheKey = this.cache.generateKey(url, params);
    
    // Check cache first
    if (!options.bypassCache) {
      const cached = this.cache.get(cacheKey);
      if (cached) {
        return cached;
      }
    }

    try {
      const queryString = new URLSearchParams(params).toString();
      const fullUrl = `${url}${queryString ? '?' + queryString : ''}`;
      
      const response = await fetch(fullUrl, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          ...options.headers
        }
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      // Cache successful responses
      if (options.cacheTtl !== 0) {
        this.cache.set(cacheKey, data, options.cacheTtl);
      }

      return data;
    } catch (error) {
      console.error('API request failed:', error);
      throw error;
    }
  }

  // POST request (typically not cached)
  async post(endpoint, data, options = {}) {
    const url = `${this.baseURL}${endpoint}`;
    
    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...options.headers
        },
        body: JSON.stringify(data)
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('API request failed:', error);
      throw error;
    }
  }

  // PUT request
  async put(endpoint, data, options = {}) {
    const url = `${this.baseURL}${endpoint}`;
    
    try {
      const response = await fetch(url, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          ...options.headers
        },
        body: JSON.stringify(data)
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('API request failed:', error);
      throw error;
    }
  }

  // DELETE request
  async delete(endpoint, options = {}) {
    const url = `${this.baseURL}${endpoint}`;
    
    try {
      const response = await fetch(url, {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json',
          ...options.headers
        }
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('API request failed:', error);
      throw error;
    }
  }

  // Invalidate cache for specific endpoint
  invalidateCache(endpoint, params = {}) {
    const url = `${this.baseURL}${endpoint}`;
    const cacheKey = this.cache.generateKey(url, params);
    this.cache.remove(cacheKey);
  }

  // Get cache statistics
  getCacheStats() {
    return this.cache.getStats();
  }

  // Clear all cache
  clearCache() {
    this.cache.clear();
  }
}

// Create global API client instance
const apiClient = new CachedAPIClient();

// Export for use in components
export default apiClient;
export { ClientCache, CachedAPIClient };