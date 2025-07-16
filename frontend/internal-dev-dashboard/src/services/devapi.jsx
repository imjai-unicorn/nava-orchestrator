// frontend/internal-dev-dashboard/src/services/devApi.js
// Internal Developer API Client - FIXED VERSION

class DevApiClient {
    constructor() {
        // ✅ FIXED: Use correct NAVA backend URL
        this.baseUrl = 'http://localhost:8005';
        this.timeout = 30000;
        
        console.log('🔧 DevApiClient initialized with baseUrl:', this.baseUrl);
    }
    
    async healthCheck() {
        try {
            // ✅ FIXED: Use correct endpoint path
            const response = await fetch(`${this.baseUrl}/health`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                },
            });
            
            if (response.ok) {
                const data = await response.json();
                console.log('✅ Backend health check passed:', data);
                return data;
            }
        } catch (error) {
            console.log('❌ Backend health check failed:', error.message);
        }
        return false;
    }
    
    async makeRequest(endpoint, options = {}) {
        const url = `${this.baseUrl}${endpoint}`;
        
        const defaultOptions = {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
        };
        
        const requestOptions = { ...defaultOptions, ...options };
        
        try {
            console.log(`🔗 Making request to: ${url}`);
            
            const response = await fetch(url, requestOptions);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            console.log(`✅ Response received from ${endpoint}:`, data);
            return data;
            
        } catch (error) {
            console.error(`❌ Request failed for ${endpoint}:`, error.message);
            throw error;
        }
    }
    
    // ✅ FIXED: Use correct NAVA endpoints
    async getSystemHealth() {
        return await this.makeRequest('/health');
    }
    
    async getSystemStatus() {
        return await this.makeRequest('/system/status');
    }
    
    async getServiceStatus() {
        return await this.makeRequest('/services/status');
    }
    
    async getPerformanceMetrics() {
        return await this.makeRequest('/dev/performance-metrics');
    }
    
    async getBehaviorPatterns() {
        return await this.makeRequest('/dev/behavior-patterns');
    }
    
    async getModels() {
        return await this.makeRequest('/models');
    }
    
    // Chat Methods for Testing
    async sendTestChat(message, options = {}) {
        const formData = new FormData();
        formData.append('message', message);
        formData.append('user_id', 'internal-dev-team');
        formData.append('preferred_model', options.model || '');
        formData.append('debug_mode', 'true');
        
        if (options.file) {
            formData.append('file', options.file);
        }
        
        return await this.makeRequest('/chat', {
            method: 'POST',
            headers: {
                // Don't set Content-Type for FormData
            },
            body: formData
        });
    }
    
    // Test specific AI models
    async testGPT(message) {
        return await this.sendTestChat(message, { model: 'gpt' });
    }
    
    async testClaude(message) {
        return await this.sendTestChat(message, { model: 'claude' });
    }
    
    async testGemini(message) {
        return await this.sendTestChat(message, { model: 'gemini' });
    }
    
    // Quick test methods
    async runQuickTests() {
        const tests = [
            { name: 'Health Check', method: () => this.healthCheck() },
            { name: 'System Status', method: () => this.getSystemStatus() },
            { name: 'Models List', method: () => this.getModels() },
            { name: 'Simple Chat', method: () => this.sendTestChat('Hello, test') }
        ];
        
        const results = {};
        
        for (const test of tests) {
            try {
                console.log(`🧪 Running test: ${test.name}`);
                const result = await test.method();
                results[test.name] = { success: true, data: result };
            } catch (error) {
                results[test.name] = { success: false, error: error.message };
            }
        }
        
        return results;
    }
    
    // Utility Methods
    async checkConnection() {
        try {
            const health = await this.healthCheck();
            return !!health;
        } catch (error) {
            return false;
        }
    }
}

// Create and export global instance
const devApi = new DevApiClient();

export default devApi;