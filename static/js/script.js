function sendMessage() {
    const userInput = document.getElementById("user-input").value;
    
    if (userInput.trim() === "") {
        return; // Don't send an empty message
    }

    // Add user's message to the chatbox
    const chatBox = document.getElementById("messages");
    const userMessage = document.createElement("div");
    userMessage.classList.add('message', 'user-message');
    userMessage.textContent = "You: " + userInput;
    chatBox.appendChild(userMessage);
    
    // Clear input field after sending message
    document.getElementById("user-input").value = "";

    // Scroll chatbox to the bottom
    chatBox.scrollTop = chatBox.scrollHeight;

    // Display a typing indicator
    const typingIndicator = document.createElement("div");
    typingIndicator.classList.add("typing-indicator");
    typingIndicator.textContent = "AI is typing...";
    chatBox.appendChild(typingIndicator);
    
    // Send the request to the backend API
    fetch("/api/chat", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ input: userInput })
    })
    .then(response => response.json())
    .then(data => {
        // Remove the typing indicator
        chatBox.removeChild(typingIndicator);

        // Add bot's response to the chatbox
        const botMessage = document.createElement("div");
        botMessage.classList.add('message', 'bot-message');
        botMessage.textContent = "AI: " + data.response;
        chatBox.appendChild(botMessage);
        
        // Scroll chatbox to the bottom
        chatBox.scrollTop = chatBox.scrollHeight;
    })
    .catch(error => {
        // Handle any errors
        console.error("Error:", error);
        chatBox.removeChild(typingIndicator);
    });
}

// Add event listener to send button
document.getElementById("send-btn").addEventListener("click", sendMessage);

// Allow sending message by pressing Enter key
document.getElementById("user-input").addEventListener("keypress", function(event) {
    if (event.key === "Enter") {
        sendMessage();
    }
});
