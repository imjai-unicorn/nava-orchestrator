class NAVAApp {
    constructor() {
        this.apiService = new APIService();
        this.chatComponent = new ChatComponent(this.apiService);
        this.statusIndicator = document.getElementById('statusIndicator');
        
        this.init();
    }

    async init() {
        console.log('🚀 NAVA Chat App starting...');
        
        // Check API health
        await this.checkAPIHealth();
        
        // Setup periodic health checks
        setInterval(() => {
            this.checkAPIHealth();
        }, 30000); // Check every 30 seconds

        console.log('✅ NAVA Chat App ready!');
    }

    async checkAPIHealth() {
        try {
            const health = await this.apiService.checkHealth();
            this.updateStatus(health);
        } catch (error) {
            console.error('Health check failed:', error);
            this.updateStatus({ status: 'unhealthy', error: error.message });
        }
    }

    updateStatus(health) {
        const statusDot = this.statusIndicator.querySelector('.status-dot');
        const statusText = this.statusIndicator.querySelector('span:last-child');
        
        if (health.status === 'healthy') {
            statusDot.style.background = '#28a745';
            statusText.textContent = 'Connected';
            
            if (health.database === 'connected') {
                statusText.textContent = 'Connected • Database OK';
            }
        } else {
            statusDot.style.background = '#dc3545';
            statusText.textContent = 'Disconnected';
        }
    }
}

// Start the app when the page loads
document.addEventListener('DOMContentLoaded', () => {
    window.app = new NAVAApp();
});