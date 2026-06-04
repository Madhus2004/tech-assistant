// static/app.js

const messagesContainer = document.getElementById("chat-messages");
const questionInput = document.getElementById("question-input");
const sendBtn = document.getElementById("send-btn");

function scrollToBottom() {
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function appendMessage(content, type, sources = []) {
    const messageDiv = document.createElement("div");
    messageDiv.className = `message ${type}-message`;

    const contentDiv = document.createElement("div");
    contentDiv.className = "message-content";
    contentDiv.innerHTML = content.replace(/\n/g, "<br>");
    messageDiv.appendChild(contentDiv);

    // Show source tags for assistant messages
    if (type === "assistant" && sources.length > 0) {
        const sourcesDiv = document.createElement("div");
        sourcesDiv.className = "message-sources";
        sources.forEach(source => {
            const tag = document.createElement("span");
            tag.className = "source-tag";
            tag.textContent = source.replace(/_/g, " ");
            sourcesDiv.appendChild(tag);
        });
        messageDiv.appendChild(sourcesDiv);
    }

    messagesContainer.appendChild(messageDiv);
    scrollToBottom();
    return messageDiv;
}

function showThinking() {
    const div = document.createElement("div");
    div.className = "message assistant-message thinking";
    div.id = "thinking-indicator";
    div.innerHTML = `<div class="message-content">Searching documents...</div>`;
    messagesContainer.appendChild(div);
    scrollToBottom();
}

function removeThinking() {
    const indicator = document.getElementById("thinking-indicator");
    if (indicator) indicator.remove();
}

async function sendQuestion() {
    const question = questionInput.value.trim();
    if (!question) return;

    // Disable input while waiting
    questionInput.value = "";
    questionInput.style.height = "auto";
    sendBtn.disabled = true;

    // Show user message
    appendMessage(question, "user");

    // Show thinking indicator
    showThinking();

    try {
        const response = await fetch("/ask", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question })
        });

        const data = await response.json();
        removeThinking();

        if (data.error) {
            appendMessage(`Error: ${data.error}`, "assistant");
        } else {
            appendMessage(data.answer, "assistant", data.sources);
        }

    } catch (error) {
        removeThinking();
        appendMessage("Network error. Please try again.", "assistant");
    }

    sendBtn.disabled = false;
    questionInput.focus();
}

function askSuggestion(btn) {
    questionInput.value = btn.textContent.trim();
    sendQuestion();
}

function handleKeyDown(event) {
    // Send on Enter, new line on Shift+Enter
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        sendQuestion();
    }
}

// Auto-resize textarea as user types
questionInput.addEventListener("input", function () {
    this.style.height = "auto";
    this.style.height = Math.min(this.scrollHeight, 120) + "px";
});