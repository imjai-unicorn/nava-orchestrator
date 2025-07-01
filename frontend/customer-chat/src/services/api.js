 
class APIService {
    constructor() {
        this.baseURL = 'http://localhost:8001';
        this.conversationId = null;
    }

    async checkHealth() {
        try {
            const response = await fetch(`${this.baseURL}/health`);
            const data = await response.json();
            return data;
        } catch (error) {
            console.error('Health check failed:', error);
            return { status: 'unhealthy', error: error.message };
        }
    }

    async sendMessage(message) {
        try {
            const requestBody = {
                message: message
            };

            if (this.conversationId) {
                requestBody.conversation_id = this.conversationId;
            }

            const response = await fetch(`${this.baseURL}/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestBody)
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            
            // Store conversation ID for future messages
            if (data.conversation_id) {
                this.conversationId = data.conversation_id;
            }

            return data;
        } catch (error) {
            console.error('Send message failed:', error);
            throw error;
        }
    }
}

// Export for use in other files
window.APIService = APIService;