 
class ChatComponent {
    constructor(apiService) {
        this.apiService = apiService;
        this.chatMessages = document.getElementById('chatMessages');
        this.messageInput = document.getElementById('messageInput');
        this.sendButton = document.getElementById('sendButton');
        this.loadingIndicator = document.getElementById('loadingIndicator');
        this.decisionContent = document.getElementById('decisionContent');
        
        this.setupEventListeners();
    }

    setupEventListeners() {
        // Send button click
        this.sendButton.addEventListener('click', () => {
            this.sendMessage();
        });

        // Enter key press
        this.messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // Auto-resize input
        this.messageInput.addEventListener('input', () => {
            this.updateSendButton();
        });
    }

    updateSendButton() {
        const hasText = this.messageInput.value.trim().length > 0;
        this.sendButton.disabled = !hasText;
    }

    async sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message) return;

        // Disable input while processing
        this.setLoading(true);
        
        // Add user message to chat
        this.addMessage(message, 'user');
        
        // Clear input
        this.messageInput.value = '';
        this.updateSendButton();

        try {
            // Send to API
            const response = await this.apiService.sendMessage(message);
            
            // Add assistant response
            this.addMessage(response.response, 'assistant', {
                model: response.model_used,
                confidence: response.confidence
            });

            // Update decision panel
            this.updateDecisionPanel(response);

        } catch (error) {
            console.error('Error sending message:', error);
            this.addMessage(
                '❌ ขออภัย เกิดข้อผิดพลาดในการประมวลผล กรุณาลองใหม่อีกครั้ง', 
                'assistant',
                { error: true }
            );
        } finally {
            this.setLoading(false);
        }
    }

    addMessage(content, role, meta = {}) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message-bubble ${role}`;

        let messageHTML = `<div class="message-content">${content}</div>`;

        if (role === 'assistant' && !meta.error) {
            messageHTML += `
                <div class="message-meta">
                    <div class="model-badge">${meta.model || 'unknown'}</div>
                    <div class="confidence">${Math.round((meta.confidence || 0) * 100)}% confidence</div>
                </div>
            `;
        }

        messageDiv.innerHTML = messageHTML;

        // Remove welcome message if this is the first real message
        const welcomeMessage = this.chatMessages.querySelector('.welcome-message');
        if (welcomeMessage && role === 'user') {
            welcomeMessage.remove();
        }

        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
    }

    updateDecisionPanel(response) {
        const reasoning = response.reasoning || {};
        
        this.decisionContent.innerHTML = `
            <div class="decision-item">
                <div class="decision-label">Selected Model</div>
                <div class="model-display">${response.model_used}</div>
            </div>

            <div class="decision-item">
                <div class="decision-label">Confidence Score</div>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${response.confidence * 100}%"></div>
                </div>
                <div class="confidence-text">${Math.round(response.confidence * 100)}% confident in this selection</div>
            </div>

            <div class="decision-item">
                <div class="decision-label">Detected Category</div>
                <div class="model-display">${reasoning.category || 'unknown'}</div>
            </div>

            ${reasoning.keywords_found && reasoning.keywords_found.length > 0 ? `
                <div class="decision-item">
                    <div class="decision-label">Keywords Found</div>
                    <div class="keywords-list">
                        ${reasoning.keywords_found.map(keyword => 
                            `<span class="keyword-tag">${keyword}</span>`
                        ).join('')}
                    </div>
                </div>
            ` : ''}

            ${reasoning.scores ? `
                <div class="decision-item">
                    <div class="decision-label">Category Scores</div>
                    <div style="font-size: 0.875rem; color: #6c757d;">
                        ${Object.entries(reasoning.scores).map(([category, score]) =>
                            `<div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                                <span>${category}:</span>
                                <span>${score}</span>
                            </div>`
                        ).join('')}
                    </div>
                </div>
            ` : ''}
        `;
    }

    setLoading(loading) {
        this.messageInput.disabled = loading;
        this.sendButton.disabled = loading;
        this.loadingIndicator.style.display = loading ? 'flex' : 'none';
    }

    scrollToBottom() {
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }
}

// Export for use in other files
window.ChatComponent = ChatComponent;