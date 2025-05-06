let isFirstMessage = true;

document.addEventListener('DOMContentLoaded', function() {
  const messagesContainer = document.querySelector('.messages');

  // Clear the messages container when the page loads
  messagesContainer.innerHTML = '';
});

// Send a message to the backend and append the response
function sendMessage(userMessage) {
  const messagesContainer = document.querySelector('.messages');
  const userInputField = document.getElementById('user-input');
  
  // Ensure that placeholder text is not sent as the user input
  if (userInputField.value.trim() === userInputField.placeholder) {
    return; // Do nothing if the user hasn't typed anything yet (i.e., the placeholder is still there)
  }

  // Append user message to the chat (this is the user's input)
  appendMessage(userMessage, 'user-message');

  // Send the user message to the backend API
  fetch('/api/chat', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ input: userMessage })
  })
  .then(response => response.json())
  .then(data => {
    // Append the response from the bot
    const botResponse = data.response;
    appendMessage(botResponse, 'bot-message');
  })
  .catch(error => {
    console.error('Error:', error);
  });
}

// Append a message to the chat (either user or bot)
function appendMessage(content, messageType) {
  const messagesContainer = document.querySelector('.messages');
  
  const message = document.createElement('div');
  message.className = `message ${messageType}`;
  message.textContent = content;

  messagesContainer.appendChild(message);
  messagesContainer.scrollTop = messagesContainer.scrollHeight; // Auto-scroll to bottom
}

// Event listener for Send button
document.getElementById('send-btn').addEventListener('click', function() {
  const userInput = document.getElementById('user-input').value;
  if (userInput.trim() !== "" && userInput !== document.getElementById('user-input').placeholder) {
    sendMessage(userInput);
    document.getElementById('user-input').value = ''; // Clear input field
  }
});

// Event listener for Enter key press
document.getElementById('user-input').addEventListener('keypress', function(event) {
  if (event.key === 'Enter') {
    const userInput = document.getElementById('user-input').value;
    if (userInput.trim() !== "" && userInput !== document.getElementById('user-input').placeholder) {
      sendMessage(userInput);
      document.getElementById('user-input').value = ''; // Clear input field
    }
  }
});
